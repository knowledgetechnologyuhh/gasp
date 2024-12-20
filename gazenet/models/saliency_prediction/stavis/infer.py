import re
import os

import torch
import numpy as np
import torch.backends.cudnn as cudnn

from gazenet.utils.registrar import *
from gazenet.utils.sample_processors import InferenceSampleProcessor

import gazenet.models.saliency_prediction.stavis.model as stavis_model
from gazenet.models.saliency_prediction.stavis.generator import load_video_frames, get_wav_features, normalize_data

MODEL_PATHS = {
    "stavis_audvis": os.path.join("gazenet", "models", "saliency_prediction", "stavis", "checkpoints", "pretrained_stavis_orig", "stavis_audiovisual", "audiovisual_split1_save_60.pth"),
    "stavis_vis": os.path.join("gazenet", "models", "saliency_prediction", "stavis", "checkpoints", "pretrained_stavis_orig", "stavis_visual_only", "visual_split1_save_60.pth")}

INP_IMG_WIDTH = 112
INP_IMG_HEIGHT = 112
INP_IMG_MEAN = (110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0)
INP_IMG_STD = (38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0)
# IMG_MEAN = [0,0,0]
# IMG_STD = [1,1,1]
# AUD_MEAN = [114.7748 / 255.0, 107.7354 / 255.0, 99.4750 / 255.0]
FRAMES_LEN = 16


@InferenceRegistrar.register
class STAViSInference(InferenceSampleProcessor):

    def __init__(self, weights_file=None, w_size=16, audiovisual=False,
                 frames_len=FRAMES_LEN,
                 inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT, inp_img_mean=INP_IMG_MEAN, inp_img_std=INP_IMG_STD,
                 device="cuda:0", width=None, height=None, **kwargs):
        self.short_name = "stavis"
        self._device = device

        self.frames_len = frames_len
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        if weights_file is None:
            if audiovisual:
                weights_file = MODEL_PATHS['stavis_audvis']
            else:
                weights_file = MODEL_PATHS['stavis_vis']

        super().__init__(width=width, height=height, w_size=w_size, **kwargs)

        # load the model
        self.model = stavis_model.resnet50(shortcut_type="B", sample_size=inp_img_width, sample_duration=frames_len,
                                           audiovisual=audiovisual)
        if weights_file in MODEL_PATHS.keys():
            weights_file = MODEL_PATHS[weights_file]
        self.model.load_model(weights_file=weights_file, device=device)
        print("STAViS model loaded from", weights_file)
        self.model = self.model.to(device)
        # cudnn.benchmarks = True
        # self.model.eval()

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    video_frames_list, hann_audio_frames, valid_audio_frames_len=None, source_frames_idxs=None, **kwargs):
        if valid_audio_frames_len is None:
            valid_audio_frames_len = self.frames_len
        audio_data = hann_audio_frames.to(self._device)
        audio_data = torch.unsqueeze(audio_data, 0)
        frames_idxs = range(len(grouped_video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for f_idx, frame_id in enumerate(frames_idxs):
            info = {"frame_detections_" + self.short_name: {
                "saliency_maps": [],  "video_saliency_maps": [], "audio_saliency_maps": [], # detected
            }}
            video_frames_tensor = load_video_frames(video_frames_list[:frame_id+1],
                                                    frame_id+1,
                                                    valid_audio_frames_len,
                                                    img_width=self.inp_img_width, img_height=self.inp_img_height,
                                                    img_mean=self.inp_img_mean, img_std=self.inp_img_std,
                                                    frames_len=self.frames_len)
            with torch.no_grad():
                video_frames = video_frames_tensor.to(self._device)
                video_frames = torch.unsqueeze(video_frames, 0)
                prediction = self.model(video_frames, audio_data)

                prediction_l = prediction["sal"][-1]
                prediction_l = torch.sigmoid(prediction_l)
                saliency = prediction_l.cpu().data.numpy()
                saliency = np.squeeze(saliency)
                saliency = normalize_data(saliency)
                info["frame_detections_" + self.short_name]["saliency_maps"].append((saliency, -1))
                info_list[frame_id].update(**info)

        kept_data = self._keep_extracted_frames_data(source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

    def preprocess_frames(self, video_frames_list=None, hann_audio_frames=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        pad = features["preproc_pad_len"]
        lim = features["preproc_lim_len"]
        if video_frames_list is not None:
            video_frames_list = list(video_frames_list)
            features["video_frames_list"] = video_frames_list[:lim] + [video_frames_list[lim]] * pad
        if hann_audio_frames is not None:
            features["hann_audio_frames"], features["valid_audio_frames_len"] = \
                get_wav_features(list(hann_audio_frames), self.frames_len, frames_len=self.frames_len)
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
                    frame_transformed = plotter.plot_color_map(saliency_map, color_map=color_map)
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
