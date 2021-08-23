import os
from itertools import zip_longest

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from gazenet.utils.registrar import *
from gazenet.models.gaze_following.videogaze.model import VideoGaze
from gazenet.utils.sample_processors import InferenceSampleProcessor

MODEL_PATHS = {
    "videogaze": os.path.join("gazenet", "models", "gaze_following", "videogaze", "checkpoints", "pretrained_videogaze_orig", "model.pth.tar")}

INP_IMG_WIDTH = 227
INP_IMG_HEIGHT = 227
TRG_IMG_SIDE = 20


@InferenceRegistrar.register
class VideoGazeInference(InferenceSampleProcessor):
    def __init__(self, weights_file=MODEL_PATHS['videogaze'],
                 batch_size=1, w_fps=30, w_size=2, 
                 trg_img_side=TRG_IMG_SIDE, inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, **kwargs)
        self.short_name = "vidgaze"
        # the original implementation skips frames
        # self.w_fps = w_fps
        # we skip one frame at a time
        self.w_fps = 1
        self._device = device

        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.trg_img_side = trg_img_side
        
        # load the model
        self.model = VideoGaze(batch_size=batch_size, side=trg_img_side)
        checkpoint = torch.load(weights_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("VideoGaze model loaded from", weights_file)
        self.model.to(device)
        cudnn.benchmark = True

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    video_frames_list, faces_locations, source_frames_idxs=None, **kwargs):
        trans = transforms.ToTensor()

        target_frame = torch.FloatTensor(self.w_size, 3, self.inp_img_width, self.inp_img_height)
        target_frame = target_frame.to(self._device)

        eyes = torch.zeros(self.w_size, 3)
        eyes = eyes.to(self._device)

        # info_list = []
        # initialize the info_list with the info structure since target sal. maps can appear in frames other than curr.
        for f_idx in range(len(info_list)):
            info_list[f_idx].update(**{"frame_detections_" + self.short_name:
                {
                    "saliency_maps": [],  # detected
                    "h_bboxes": []  # processed
                }})

        frames_idxs = range(len(video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for frame_id in frames_idxs:
            # info = {"frame_detections_" + self.short_name: {
            #     "saliency_maps": [],  # detected
            #     "h_bboxes": []  # processed
            # }}

            # print('Processing of frame %d out of %d' % (i, len(video_frames_list)))

            # avoid the problems with the video limit
            # if self.w_fps * (self.w_size - 1) // 2 < frame_id < (len(video_frames_list) - self.w_fps * (self.w_size - 1) // 2) \
            #         and len(faces_locations) > 0:
            if len(faces_locations) > 0:
                # read the image
                img = video_frames_list[frame_id]
                if img is None:
                    continue
                h, w, c = img.shape

                # resize image
                img_resized = cv2.resize(img, (self.inp_img_width, self.inp_img_height))
                for id, face_local in enumerate(faces_locations[frame_id]):
                    if not face_local:
                        continue
                    (top, right, bottom, left) = face_local
                    info_list[frame_id]["frame_detections_" + self.short_name]["h_bboxes"].append((left, top, right, bottom, id))
                    # crop face image
                    crop_img_face = img[top:bottom, left:right]
                    crop_img_face = cv2.resize(crop_img_face, (self.inp_img_width, self.inp_img_height))

                    # compute the center of the head and estimate the eyes location
                    eyes[:, 0] = (right + left) / (2 * w)
                    eyes[:, 1] = (top + bottom) / (2 * h)

                    # fill the tensors for the exploring window. Face and source frame are the same
                    source_frame = trans(img_resized).view(1, 3, self.inp_img_width, self.inp_img_height)
                    face_frame = trans(crop_img_face).view(1, 3, self.inp_img_width, self.inp_img_height)
                    for j in range(self.w_size - 1):
                        trans_im = trans(img_resized).view(1, 3, self.inp_img_width, self.inp_img_height)
                        source_frame = torch.cat((source_frame, trans_im), 0)
                        crop_img = trans(crop_img_face).view(1, 3, self.inp_img_width, self.inp_img_height)
                        face_frame = torch.cat((face_frame, crop_img), 0)

                    # fill the targets for the exploring window.
                    for j in range(self.w_size):
                        # target_im = video_frames_list[frame_id + self.w_fps * (j - ((self.w_size - 1) // 2))]
                        target_im = video_frames_list[j]
                        target_im = cv2.resize(target_im, (self.inp_img_width, self.inp_img_height))
                        target_im = trans(target_im)
                        target_frame[j, :, :, :] = target_im

                    # run the model
                    source_frame = source_frame.to(self._device, non_blocking=True)
                    target_frame = target_frame.to(self._device, non_blocking=True)
                    face_frame = face_frame.to(self._device, non_blocking=True)
                    eyes = eyes.to(self._device, non_blocking=True)
                    source_frame_var = torch.autograd.Variable(source_frame)
                    target_frame_var = torch.autograd.Variable(target_frame)
                    face_frame_var = torch.autograd.Variable(face_frame)
                    eyes_var = torch.autograd.Variable(eyes)
                    output, sigmoid = self.model(source_frame_var, target_frame_var, face_frame_var, eyes_var)

                    # recover the data from the variables
                    sigmoid = sigmoid.data
                    output = output.data

                    # pick the maximum value for the frame selection
                    v, ids = torch.sort(sigmoid, dim=0, descending=True)
                    index_target = ids[0, 0]

                    # pick the video_frames_list corresponding to the maximum value
                    # target_im = frame_list[i + w_fps * (index_target - ((N - 1) // 2))].copy()
                    output_target = output[index_target, :, :, :].view(self.trg_img_side, self.trg_img_side).cpu().numpy()

                    # compute the gaze location
                    # heatmaps += output_target
                    # info["frame_detections_" + self.short_name]["saliency_maps"].append((output_target, id))
                    # info_list[i + w_fps * (index_target.cpu().numpy() - ((self.w_size - 1) // 2))][
                    info_list[index_target.cpu().numpy()][
                        "frame_detections_" + self.short_name]["saliency_maps"].append((output_target, id))
                # info_list.append(info)
        # kept_data = self.keep_extracted_frames_data(None, grabbed_video_list, grouped_video_frames_list,
        #                                             grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list

    def preprocess_frames(self, video_frames_list=None, faces_locations=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        pad = features["preproc_pad_len"]
        lim = features["preproc_lim_len"]
        if video_frames_list is not None:
            video_frames_list = list(video_frames_list)
            features["video_frames_list"] = video_frames_list[:lim] + [video_frames_list[lim]] * pad
        if faces_locations is not None:
            faces_locations = list(faces_locations)
            features["faces_locations"] = faces_locations[:lim] + [faces_locations[lim]] * pad
        return features

    def annotate_frame(self, input_data, plotter,
                       show_det_saliency_map=True,
                       show_det_gaze_target=True,
                       show_det_head_bbox=True,
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = input_data

        properties = {"show_det_saliency_map": (show_det_saliency_map, "toggle", (True, False)),
                      "show_det_gaze_target": (show_det_gaze_target, "toggle", (True, False)),
                      "show_det_head_bbox": (show_det_head_bbox, "toggle", (True, False))}

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": grouped_video_frames["PLOT"] + [["det_source_" + self.short_name,
                                                                         "det_transformed_" + self.short_name]],
                                "det_source_" + self.short_name: grouped_video_frames["captured"],
                                "det_transformed_" + self.short_name: grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}
        if grabbed_video:
            collated_saliency_maps = np.zeros((self.trg_img_side, self.trg_img_side))
            if show_det_head_bbox or show_det_saliency_map or show_det_gaze_target:
                for h_bbox, saliency_map in zip_longest(info["frame_detections_" + self.short_name]["h_bboxes"],
                                                        info["frame_detections_" + self.short_name]["saliency_maps"]):
                    if h_bbox is not None:
                        xmin_h_bbox, ymin_h_bbox, xmax_h_bbox, ymax_h_bbox, participant_id = h_bbox
                        if show_det_head_bbox:
                            frame_source = plotter.plot_bbox(grouped_video_frames["det_source_" + self.short_name],
                                                             (xmin_h_bbox, ymin_h_bbox),
                                                             (xmax_h_bbox, ymax_h_bbox), color_id=participant_id)
                            grouped_video_frames["det_source_" + self.short_name] = frame_source
                    if saliency_map is not None:
                        if show_det_saliency_map:
                            collated_saliency_maps += saliency_map[0]
                        if show_det_gaze_target:
                            map = np.reshape(saliency_map[0], (self.trg_img_side * self.trg_img_side))
                            int_class = np.argmax(map)
                            x_class = int_class % self.trg_img_side
                            y_class = (int_class - x_class) // self.trg_img_side
                            y_float = y_class / self.trg_img_side
                            x_float = x_class / self.trg_img_side
                            x_point = np.floor(x_float * grouped_video_frames["det_transformed_" + self.short_name].shape[1]).astype(np.int32)
                            y_point = np.floor(y_float * grouped_video_frames["det_transformed_" + self.short_name].shape[0]).astype(np.int32)
                            frame_transformed = plotter.plot_point(grouped_video_frames["det_transformed_" + self.short_name],
                                                                   (x_point, y_point), color_id=saliency_map[1], radius=10)
                            grouped_video_frames["det_transformed_" + self.short_name] = frame_transformed
                if show_det_saliency_map:
                    frame_transformed = plotter.plot_color_map(np.uint8(255 * collated_saliency_maps),
                                                               color_map=color_map)
                    if enable_transform_overlays:
                        frame_transformed = plotter.plot_alpha_overlay(grouped_video_frames["det_transformed_" + self.short_name],
                                                                       frame_transformed, alpha=0.4)
                    grouped_video_frames["det_transformed_" + self.short_name] = frame_transformed

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties


