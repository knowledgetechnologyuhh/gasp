import random
import cv2
import re
import os

import torch
import numpy as np

from gazenet.utils.registrar import *
from gazenet.models.saliency_prediction.unisal.generator import load_video_frames
from gazenet.models.saliency_prediction.unisal.model import UNISAL
from gazenet.utils.sample_processors import InferenceSampleProcessor

from gazenet.models.saliency_prediction.unisal.generator import smooth_sequence


MODEL_PATHS = {
    "unisal": os.path.join("gazenet", "models", "saliency_prediction", "unisal", "checkpoints", "pretrained_unisal_orig", "model.pth")}

INP_IMG_WIDTH = 384
INP_IMG_HEIGHT = 288
TRG_IMG_WIDTH = 640
TRG_IMG_HEIGHT = 480
INP_IMG_MEAN = (0.485, 0.456, 0.406)
INP_IMG_STD = (0.229, 0.224, 0.225)
FRAMES_LEN = 12


@InferenceRegistrar.register
class UNISALInference(InferenceSampleProcessor):

    def __init__(self, weights_file=MODEL_PATHS['unisal'], w_size=12,
                 frames_len=FRAMES_LEN, trg_img_width=TRG_IMG_WIDTH, trg_img_height=TRG_IMG_HEIGHT,
                 inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT, inp_img_mean=INP_IMG_MEAN, inp_img_std=INP_IMG_STD,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, **kwargs)
        self.short_name = "unisal"
        self._device = device

        self.frames_len = frames_len
        self.trg_img_width = trg_img_width
        self.trg_img_height = trg_img_height
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        # load the model
        self.model = UNISAL()
        self.model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))
        print("UNISAL model loaded from", weights_file)
        self.model = self.model.to(device)
        self.model.eval()

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list,
                    info_list, properties_list,
                    video_frames_list, source_frames_idxs=None, smooth_method="med41", **kwargs):

        frames_idxs = range(len(grouped_video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        h0 = [None]
        model_kwargs = {
            'source': ("DHF1K"), #"eval",
            'target_size': (self.trg_img_height, self.trg_img_width)}

        for f_idx, frame_id in enumerate(frames_idxs):
            info = {"frame_detections_" + self.short_name: {
                "saliency_maps": [],  # detected
            }}
            video_frames_tensor = load_video_frames(video_frames_list[:frame_id + 1],
                                                    frame_id + 1,
                                                    img_width=self.inp_img_width, img_height=self.inp_img_height,
                                                    img_mean=self.inp_img_mean, img_std=self.inp_img_std,
                                                    frames_len=self.frames_len)
            video_frames = video_frames_tensor.to(self._device)
            video_frames = torch.unsqueeze(video_frames, 0)
            with torch.no_grad():
                final_prediction, _ = self.model( # final_prediction, h0 = self.model(
                    video_frames, h0=h0, return_hidden=True,
                    **model_kwargs)


            # get the visual feature maps
            for prediction, prediction_name in zip([final_prediction], ["saliency_maps"]):
                saliency = prediction.cpu()
                if smooth_method is not None:
                    saliency = saliency.numpy()
                    saliency = smooth_sequence(saliency, smooth_method)
                    saliency = torch.from_numpy(saliency).float()

                # for _, smap in enumerate(torch.unbind(saliency, dim=1)):
                smap = saliency[:, frame_id, ::]
                smap = smap.exp()
                smap = torch.squeeze(smap)
                smap = smap.data.cpu().numpy()
                smap = smap / np.amax(smap)
                info["frame_detections_" + self.short_name][prediction_name].append((smap, -1))
            info_list[frame_id].update(**info)

        kept_data = self._keep_extracted_frames_data(source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

    def preprocess_frames(self, video_frames_list=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        pad = features["preproc_pad_len"]
        lim = features["preproc_lim_len"]
        if video_frames_list is not None:
            video_frames_list = list(video_frames_list)
            features["video_frames_list"] = video_frames_list[:lim] + [video_frames_list[lim]] * pad
        return features

    def annotate_frame(self, input_data, plotter,
                       show_det_saliency_map=True,
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = input_data

        properties = {**properties, "show_det_saliency_map": (show_det_saliency_map, "toggle", (True, False))}

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": grouped_video_frames["PLOT"] + [["det_source_" + self.short_name,
                                                                         "det_transformed_" + self.short_name]],
                                "det_source_" + self.short_name: grouped_video_frames["captured"],
                                "det_transformed_" + self.short_name: grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}

        for saliency_map_name, frame_name in zip(["saliency_maps"], [""]):
            if grabbed_video:
                if show_det_saliency_map:
                    saliency_map = info["frame_detections_" + self.short_name][saliency_map_name][0][0]
                    frame_transformed = plotter.plot_color_map(np.uint8(255 * saliency_map), color_map=color_map)
                    if enable_transform_overlays:
                        frame_transformed = plotter.plot_alpha_overlay(grouped_video_frames["det_transformed_" +
                                                                                            frame_name + self.short_name],
                                                                       frame_transformed, alpha=0.4)
                    else:
                        frame_transformed = plotter.resize(frame_transformed,
                                                           height=grouped_video_frames["det_transformed_" + frame_name +
                                                                                       self.short_name].shape[0],
                                                           width=grouped_video_frames["det_transformed_" + frame_name +
                                                                                      self.short_name].shape[1])
                    grouped_video_frames["det_transformed_" + frame_name + self.short_name] = frame_transformed
        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties
