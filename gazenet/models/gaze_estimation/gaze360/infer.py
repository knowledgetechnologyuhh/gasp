import os

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from gazenet.utils.registrar import *
from gazenet.models.gaze_estimation.gaze360.generator import spherical_to_cartesian, spherical_to_compatible_form, pre_process_input_image
from gazenet.models.gaze_estimation.gaze360.model import GazeLSTM, DataParallel
from gazenet.utils.sample_processors import InferenceSampleProcessor
from gazenet.utils.helpers import spherical_to_euler

MODEL_PATHS = {
    "gaze360": os.path.join("gazenet", "models", "gaze_estimation", "gaze360", "checkpoints", "pretrained_gaze360_orig", "model.pth.tar")}

INP_IMG_WIDTH = 224
INP_IMG_HEIGHT = 224
INP_IMG_MEAN = (0.485, 0.456, 0.406)
INP_IMG_STD = (0.229, 0.224, 0.225)


@InferenceRegistrar.register
class Gaze360Inference(InferenceSampleProcessor):
    def __init__(self, weights_file=MODEL_PATHS['gaze360'], w_fps=30, inp_img_width=INP_IMG_WIDTH,
                 inp_img_height=INP_IMG_HEIGHT, inp_img_mean=INP_IMG_MEAN, inp_img_std=INP_IMG_STD,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, **kwargs)

        self.short_name = "gaze360"
        # the original implementation skips an 8th of a frame when scanning surrounding frames
        # self.w_fps_div = max(int(w_fps // 8), 1)
        # we skip one frame at a time
        self.w_fps_div = 1
        self._device = device

        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std
        # extra grouping list
        self.faces_locations = []

        # load the model
        self.model = GazeLSTM()
        model = DataParallel(self.model).to(device)
        if weights_file in MODEL_PATHS.keys():
            weights_file = MODEL_PATHS[weights_file]
        model.load_model(weights_file=weights_file)
        print("Gaze360 model loaded from", weights_file)
        model.eval()


    # adapted from: https://colab.research.google.com/drive/1AUvmhpHklM9BNt0Mn5DjSo3JRuqKkU4y#scrollTo=FKESCskkymbs
    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    video_frames_list, faces_locations, source_frames_idxs=None, **kwargs):
        frames_idxs = range(len(video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for frame_id in frames_idxs:
            info = {"frame_detections_" + self.short_name: {
                "CART_gaze_poses": [],  # detected
                "SPHERE_gaze_poses": [],  # detected
                "h_bboxes": []  # processed
            }}
            # read image
            input_image = torch.zeros(7, 3, self.inp_img_width, self.inp_img_height)

            for id, face_local in enumerate(faces_locations[frame_id]):
                if not face_local:
                    continue
                (top, right, bottom, left) = face_local
                info["frame_detections_" + self.short_name]["h_bboxes"].append((left, top, right, bottom, id))

                count = 0
                for j in range(frame_id - 3 * self.w_fps_div, frame_id + 4 * self.w_fps_div, self.w_fps_div):
                    if (j < 0 or j >= len(faces_locations)) and video_frames_list[frame_id] is not None:
                        face_img = video_frames_list[frame_id].copy()
                    else:
                        # TODO (fabawi): this approach will not work if we intend on supporting visual object tracking
                        if id < len(faces_locations[j]) and faces_locations[j] is not None and video_frames_list[j] is not None:
                            face_img = video_frames_list[j].copy()
                            face_local = faces_locations[j][id]
                        elif faces_locations[j] is None:
                            face_local = False
                            continue
                        elif video_frames_list[frame_id] is None:
                           continue
                        else:
                            face_img = video_frames_list[frame_id].copy()

                    (top, right, bottom, left) = face_local
                    # crop face image
                    crop_img_face = face_img[top:bottom, left:right]

                    # fill the images
                    input_image[count, :, :, :] = pre_process_input_image(crop_img_face,
                                                                          self.inp_img_width, self.inp_img_height,
                                                                          self.inp_img_height, self.inp_img_std)
                    count = count + 1

                # bbox, eyes = tracking_id[i][id_t]
                # bbox = np.asarray(bbox).astype(int)
                output_gaze, _ = self.model(input_image.view(1, 7, 3,
                                                             self.inp_img_width, self.inp_img_height).to(self._device))
                gaze = spherical_to_cartesian(output_gaze).detach().numpy()
                gaze = gaze.reshape((-1))
                info["frame_detections_" + self.short_name]["CART_gaze_poses"].append((gaze[0], gaze[1], gaze[2], id))
                gaze = spherical_to_compatible_form(output_gaze).detach().numpy()
                gaze = gaze.reshape((-1))
                info["frame_detections_" + self.short_name]["SPHERE_gaze_poses"].append((gaze[0], gaze[1], gaze[2], id))
            info_list[frame_id].update(**info)

        kept_data = self._keep_extracted_frames_data(source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

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
                       show_det_gaze_axis=False,
                       show_det_gaze_direction_field=True,
                       show_det_head_bbox=True,
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = input_data

        properties = {**properties, "show_det_gaze_axis": (show_det_gaze_axis, "toggle", (True, False)),
                      "show_det_gaze_direction_field": (show_det_gaze_direction_field, "toggle", (True, False)),
                      "show_det_head_bbox": (show_det_head_bbox, "toggle", (True, False))}

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": grouped_video_frames["PLOT"] + [["det_source_" + self.short_name,
                                                                         "det_transformed_" + self.short_name]],
                                "det_source_" + self.short_name: grouped_video_frames["captured"],
                                "det_transformed_" + self.short_name: grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}

        if grabbed_video:
            if show_det_head_bbox or show_det_gaze_axis or show_det_gaze_direction_field:
                for h_bbox, gaze_pose_cart, gaze_pose_sphere in zip(info["frame_detections_" + self.short_name]["h_bboxes"],
                                                                    info["frame_detections_" + self.short_name]["CART_gaze_poses"],
                                                                    info["frame_detections_" + self.short_name]["SPHERE_gaze_poses"]):
                    xmin_h_bbox, ymin_h_bbox, xmax_h_bbox, ymax_h_bbox, participant_id = h_bbox
                    if show_det_head_bbox:
                        frame_source = plotter.plot_bbox(grouped_video_frames["det_source_" + self.short_name],
                                                         (xmin_h_bbox, ymin_h_bbox),
                                                         (xmax_h_bbox, ymax_h_bbox), color_id=participant_id)
                        grouped_video_frames["det_source_" + self.short_name] = frame_source
                    if show_det_gaze_axis:
                        frame_transformed = plotter.plot_axis(grouped_video_frames["det_transformed_" + self.short_name],
                                                              (xmin_h_bbox, ymin_h_bbox),
                                                              (xmax_h_bbox, ymax_h_bbox), spherical_to_euler(gaze_pose_sphere))
                        grouped_video_frames["det_transformed_" + self.short_name] = frame_transformed
                    if show_det_gaze_direction_field:
                        gaze_orig = (xmin_h_bbox, ymin_h_bbox, 0)
                        frame_transformed = \
                            plotter.plot_conic_field(grouped_video_frames["det_transformed_" + self.short_name], gaze_orig,
                                                     gaze_pose_cart, radius_orig=1, radius_tgt=60, color_map=color_map)
                        grouped_video_frames["det_transformed_" + self.short_name] = \
                            plotter.plot_alpha_overlay(grouped_video_frames["det_transformed_" + self.short_name],
                                                       frame_transformed,
                                                       alpha=0.4 if enable_transform_overlays else 0.5)

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties
