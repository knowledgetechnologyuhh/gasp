#
# DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction
# https://arxiv.org/abs/1905.10693
# https://hrtavakoli.github.io/DAVE/
#
# Copyright by Hamed Rezazadegan Tavakoli
#

import re
import os

import torch
import numpy as np

from gazenet.utils.registrar import *
from gazenet.models.saliency_prediction.dave.generator import get_wav_features, load_video_frames
from gazenet.models.saliency_prediction.dave.model import DAVE
from gazenet.utils.sample_processors import InferenceSampleProcessor


MODEL_PATHS = {
    "dave": os.path.join("gazenet", "models", "saliency_prediction", "dave", "checkpoints", "pretrained_dave_orig", "model.pth.tar")}

INP_IMG_WIDTH = 320
INP_IMG_HEIGHT = 256
# TRG_IMG_WIDTH = 40
# TRG_IMG_HEIGHT = 32
INP_IMG_MEAN = (110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0)
INP_IMG_STD = (38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0)
FRAMES_LEN = 16


@InferenceRegistrar.register
class DAVEInference(InferenceSampleProcessor):

    def __init__(self, weights_file=MODEL_PATHS['dave'], w_size=16,
                 frames_len=FRAMES_LEN,
                 inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT, inp_img_mean=INP_IMG_MEAN, inp_img_std=INP_IMG_STD,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, **kwargs)
        self.short_name = "dave"
        self._device = device

        self.frames_len = frames_len
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        # load the model
        self.model = DAVE(frames_len=frames_len)
        if weights_file in MODEL_PATHS.keys():
            weights_file = MODEL_PATHS[weights_file]
        self.model.load_model(weights_file=weights_file, device=device)
        print("DAVE model loaded from", weights_file)
        self.model = self.model.to(device)
        self.model.eval()

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    video_frames_list, audio_features=None, valid_audio_frames_len=None, source_frames_idxs=None, **kwargs):
        if valid_audio_frames_len is None:
            valid_audio_frames_len = self.frames_len

        if audio_features is not None:
            audio_data = audio_features.to(self._device)
            audio_data = torch.unsqueeze(audio_data, 0)
        else:
            audio_data = None
        frames_idxs = range(len(grouped_video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for f_idx, frame_id in enumerate(frames_idxs):
            info = {"frame_detections_" + self.short_name: {
                "saliency_maps": [],  "video_saliency_maps": [], "audio_saliency_maps": [], # detected
                # "audio_features": []  # processed (this is inefficient since the features are the same for all frames)
            }}
            video_frames_tensor = load_video_frames(video_frames_list[:frame_id+1],
                                                    frame_id+1,
                                                    valid_audio_frames_len,
                                                    img_width=self.inp_img_width, img_height=self.inp_img_height,
                                                    img_mean=self.inp_img_mean, img_std=self.inp_img_std,
                                                    frames_len=self.frames_len)
            video_frames = video_frames_tensor.to(self._device)
            video_frames = torch.unsqueeze(video_frames, 0)
            final_prediction, video_prediction, audio_prediction = self.model(video_frames, audio_data,
                                                                        return_latent_streams=True)
            # get the visual and auditory feature maps
            # for prediction, prediction_name in zip([final_prediction, video_prediction, audio_prediction],
            #                                        ["saliency_maps", "video_saliency_maps", "audio_saliency_maps"]):
            for prediction, prediction_name in zip([final_prediction], ["saliency_maps"]):
                saliency = prediction.cpu().data.numpy()
                saliency = np.squeeze(saliency)
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
                info["frame_detections_" + self.short_name][prediction_name].append((saliency, -1))
            info_list[frame_id].update(**info)

        kept_data = self._keep_extracted_frames_data(source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

    def preprocess_frames(self, video_frames_list=None, audio_features=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        pad = features["preproc_pad_len"]
        lim = features["preproc_lim_len"]
        if video_frames_list is not None:
            video_frames_list = list(video_frames_list)
            features["video_frames_list"] = video_frames_list[:lim] + [video_frames_list[lim]] * pad
        if audio_features and audio_features is not None:
            features["audio_features"], features["valid_audio_frames_len"] = \
                get_wav_features(list(audio_features), self.frames_len, frames_len=self.frames_len)
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
        # grouped_video_frames["det_transformed_video_" + self.short_name] = grouped_video_frames["det_transformed_" + self.short_name].copy()
        # grouped_video_frames["det_transformed_audio_" + self.short_name] = grouped_video_frames["det_transformed_" + self.short_name].copy()

        # for saliency_map_name, frame_name in zip(["saliency_maps", "video_saliency_maps", "audio_saliency_maps"],
        #                                          ["", "video_", "audio_"]):
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
