import re
import os
from collections import deque

import cv2
import torch
import numpy as np

from gazenet.utils.registrar import *
from gazenet.models.gaze_prediction.spi_spp.generator import load_video_frames

from gazenet.utils.sample_processors import InferenceSampleProcessor
from gazenet.utils.helpers import mp_multivariate_gaussian


MODEL_PATHS = {
    "seqdamgmualstm": os.path.join("gazenet", "models", "saliency_prediction", "gasp", "checkpoints",
                                   "pretrained_sequencegaspdamencgmualstmconv",
                                   "SequenceGASPDAMEncGMUALSTMConv", "53ea3d5639d647fc86e3974d6e1d1719", "last_model.pt"),
    "seqdamalstmgmu": os.path.join("gazenet", "models", "gaze_prediction", "spi_spp", "checkpoints",
                                   "pretrained_sequencegaspdamencalstmgmuconv",
                                    "SequenceGASPDAMEncALSTMGMUConv", "367f5cf9a3e7451982b5201836f22c84", "last_model.pt")
}

INP_IMG_WIDTH = 120
INP_IMG_HEIGHT = 120
TRG_IMG_WIDTH = INP_IMG_WIDTH//2
TRG_IMG_HEIGHT = INP_IMG_HEIGHT//2
INP_IMG_MEAN = (110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0)
INP_IMG_STD = (38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0)
FRAMES_LEN = 10


@InferenceRegistrar.register
class SPIGASPInference(InferenceSampleProcessor):

    def __init__(self, weights_file=MODEL_PATHS['seqdamalstmgmu'], model_name="SequenceGASPDAMEncALSTMGMUConv", w_size=10,
                 frames_len=FRAMES_LEN, trg_img_width=TRG_IMG_WIDTH, trg_img_height=TRG_IMG_HEIGHT,
                 inp_img_width=INP_IMG_WIDTH, inp_img_height=INP_IMG_HEIGHT, inp_img_mean=INP_IMG_MEAN, inp_img_std=INP_IMG_STD,
                 device="cuda:0", width=None, height=None, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, **kwargs)
        self.short_name = "gasp"
        self._device = device

        self.frames_len = frames_len
        self.trg_img_width = trg_img_width
        self.trg_img_height = trg_img_height
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std
        # scan model registry
        ModelRegistrar.scan()

        # load the model
        kwargs.update(batch_size=1)
        self.model = ModelRegistrar.registry[model_name](**kwargs)
        if weights_file in MODEL_PATHS.keys():
            weights_file = MODEL_PATHS[weights_file]
        self.model.load_model(weights_file=weights_file)
        print("SPIGASP model loaded from", weights_file)
        self.model = self.model.to(device)
        self.model.eval()

        gaussian_image = mp_multivariate_gaussian(np.array([[60, 60, 10]]), 120, 120)
        gaussian_image = (gaussian_image / gaussian_image.max() * 255).astype(np.uint8)
        gaussian_image = np.expand_dims(gaussian_image, (0, -1))
        gaussian_images = np.repeat(gaussian_image, 10, axis=0)
        gaussian_images = np.repeat(gaussian_images, 3, axis=-1)
        self.prev_saliency = deque(gaussian_images, maxlen=self.frames_len)

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    previous_maps=None, compute_gate_scores=True, inp_img_names_list=None, source_frames_idxs=None, **kwargs):
        # when realtime data is passed to GASP
        if previous_maps is not None:
            # Prepend to prev_map according to form

            grouped_video_frames_list = list(previous_maps.values())
            grabbed_video_list = [True]*len(grouped_video_frames_list)

        frames_idxs = range(len(grouped_video_frames_list)) if source_frames_idxs is None else source_frames_idxs

        for f_idx, frame_id in enumerate(frames_idxs):
            info = {"frame_detections_" + self.short_name: {
                "saliency_maps": [],  "gate_scores": []
            }}

            _queue = list(self.prev_saliency)
            for i, item in enumerate(_queue):
                grouped_video_frames_list[i]["transformed_salmap_p"] = item
            video_frames_tensor = load_video_frames(grouped_video_frames_list[:frame_id+1],
                                                    frame_id+1,
                                                    inp_img_names_list=["transformed_salmap_p"]+inp_img_names_list,
                                                    img_width=self.inp_img_width, img_height=self.inp_img_height,
                                                    img_mean=self.inp_img_mean, img_std=self.inp_img_std,
                                                    frames_len=self.frames_len)

            video_frames = video_frames_tensor.to(self._device)
            video_frames = torch.unsqueeze(video_frames, 0)
            final_prediction = self.model(video_frames)
            prediction = torch.sigmoid(final_prediction[0])
            saliency = prediction.cpu().data.numpy()
            saliency = np.squeeze(saliency)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

            info["frame_detections_" + self.short_name]["saliency_maps"].append((saliency, -1))
            if final_prediction[2] is not None and compute_gate_scores:
                gate_scores = final_prediction[2][1].cpu().data.numpy()
                gate_scores = gate_scores.mean(axis=(2,3,4) if len(gate_scores.shape) == 5 else (2,3))
                gate_scores = np.squeeze(gate_scores)
                info["frame_detections_" + self.short_name]["gate_scores"].append((gate_scores, -1))
            info_list[frame_id].update(**info)
        # Keep predicted saliency for next step
        # self.prev_saliency = info_list[]

        kept_data = self._keep_extracted_frames_data(source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

    def preprocess_frames(self, previous_data, inp_img_names_list=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        features["inp_img_names_list"] = inp_img_names_list
        previous_maps = {}
        if previous_data is None:
            previous_data = ((None, features["grouped_video_frames_list"], None, None, features["info_list"]),)

        for result in previous_data:
            if isinstance(result, tuple):
                for f_idx, frame_data in enumerate(result[4]):
                    for plot_name, plot in result[1][f_idx].items():
                        if plot_name != "PLOT" and (inp_img_names_list is None or plot_name in inp_img_names_list):
                            if f_idx in previous_maps:
                                if plot_name in previous_maps[f_idx]:
                                    previous_maps[f_idx][plot_name] = plot
                                else:
                                    previous_maps[f_idx].update(**{plot_name: plot})
                            else:
                                previous_maps[f_idx] = {plot_name: plot}
                    previous_maps[f_idx]["PLOT"] = [[]]
            features["previous_maps"] = previous_maps
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
                    saliency_transformed = plotter.plot_color_map(np.uint8(255 * saliency_map), color_map="bone")
                    saliency_transformed = plotter.resize(saliency_transformed,
                                                          height=grouped_video_frames["det_transformed_" + frame_name +
                                                                           self.short_name].shape[0],
                                                          width=grouped_video_frames["det_transformed_" + frame_name +
                                                                          self.short_name].shape[1])
                    self.prev_saliency.append(saliency_transformed)

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
