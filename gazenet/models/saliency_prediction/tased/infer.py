import re
import sys
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from scipy.ndimage.filters import gaussian_filter

from gazenet.utils.registrar import *
from gazenet.models.saliency_prediction.tased.generator import load_video_frames
from gazenet.models.saliency_prediction.tased.model import TASED_v2
from gazenet.utils.sample_processors import InferenceSampleProcessor


MODEL_PATHS = {
    "tased": os.path.join("gazenet", "models", "saliency_prediction", "tased", "checkpoints", "pretrained_tased_orig", "model.pt")}

INP_IMG_WIDTH = 384
INP_IMG_HEIGHT = 224
FRAMES_LEN = 32


@InferenceRegistrar.register
class TASEDInference(InferenceSampleProcessor):

    def __init__(self, weights_file=MODEL_PATHS['tased'], w_size=32,
                 frames_len=FRAMES_LEN, inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, **kwargs)
        self.short_name = "tased"
        self._device = device

        self.frames_len = frames_len
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height

        # load the model
        self.model = TASED_v2()
        if weights_file in MODEL_PATHS.keys():
            weights_file = MODEL_PATHS[weights_file]
        self.model.load_model(weights_file=weights_file)
        print("TASED model loaded from", weights_file)
        self.model = self.model.to(device)
        cudnn.benchmark = False
        self.model.eval()

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    video_frames_list, source_frames_idxs=None, **kwargs):

        frames_idxs = range(len(grouped_video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for f_idx, frame_id in enumerate(frames_idxs):
            info = {"frame_detections_" + self.short_name: {
                "saliency_maps": [], # detected
            }}
            video_frames_tensor = load_video_frames(video_frames_list[:frame_id+1],
                                                    frame_id+1,
                                                    img_width=self.inp_img_width, img_height=self.inp_img_height,
                                                    frames_len=self.frames_len)
            video_frames = video_frames_tensor.to(self._device)
            video_frames = torch.unsqueeze(video_frames, 0)
            with torch.no_grad():
                final_prediction = self.model(video_frames)
            # get the visual feature maps
            for prediction, prediction_name in zip([final_prediction],["saliency_maps"]):
                saliency = prediction.cpu().data[0].numpy()
                # saliency = (saliency*255.).astype(np.int)/255.
                saliency = gaussian_filter(saliency, sigma=7)
                saliency = saliency/np.max(saliency)
                info["frame_detections_" + self.short_name][prediction_name].append((saliency, -1))
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

        for saliency_map_name, frame_name in zip(["saliency_maps"],[""]):
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
