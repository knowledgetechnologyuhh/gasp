#
# GreedyGaze: This is a very simple gaze prediction module for presenting a single peak based on multiple feature maps
#   by averaging or summing the maps and producing one point
#
import re
import os

import numpy as np
from scipy.stats import gmean

from gazenet.utils.registrar import *
from gazenet.utils.sample_processors import InferenceSampleProcessor
from gazenet.utils.helpers import pixels_to_bounded_range


@InferenceRegistrar.register
class GreedyGazeInference(InferenceSampleProcessor):

    def __init__(self, width=None, height=None, w_size=16, **kwargs):
        super().__init__(width=width, height=height, w_size=w_size, **kwargs)
        self.short_name = "greedygaze"

    def infer_frame(self, grabbed_video_list, grouped_video_frames_list, grabbed_audio_list, audio_frames_list, info_list, properties_list,
                    previous_maps=None, audio_features=None, source_frames_idxs=None, available_keys=None, **kwargs):
        if available_keys is None:
            available_keys = list(previous_maps.keys())
        updated_source_frames_idxs = []
        frames_idxs = range(len(grouped_video_frames_list)) if source_frames_idxs is None else source_frames_idxs
        for f_idx, frame_id in enumerate(frames_idxs):
            if info_list[frame_id]["frame_info"]["frame_id"] in available_keys:
                updated_source_frames_idxs.append(frame_id)
            else:
                continue
            info = {"frame_detections_" + self.short_name: {
                "target_maps": [],  # detected
                "audio_features": [],  # detected
            }}
            gaze_target = previous_maps[info_list[frame_id]["frame_info"]["frame_id"]]

            info["frame_detections_" + self.short_name]["target_maps"].append(gaze_target)
            info["frame_detections_" + self.short_name]["audio_features"].append(audio_features)
            info_list[frame_id].update(**info)

        kept_data = self._keep_extracted_frames_data(updated_source_frames_idxs, grabbed_video_list, grouped_video_frames_list,
                                                     grabbed_audio_list, audio_frames_list, info_list, properties_list)
        return kept_data

    def preprocess_frames(self, previous_data, inp_img_names_list=None, audio_features=None, **kwargs):
        features = super().preprocess_frames(**kwargs)
        previous_maps = {}
        if previous_data is None:
            for f_idx, frame_data in enumerate(features["info_list"]):
                for plot_name, plot in features["grouped_video_frames_list"][f_idx].items():
                    if plot_name != "PLOT" and (inp_img_names_list is None or plot_name in inp_img_names_list):
                        if frame_data["frame_info"]["frame_id"] in previous_maps:
                            previous_maps[frame_data["frame_info"]["frame_id"]].append(plot)
                        else:
                            previous_maps[frame_data["frame_info"]["frame_id"]] = [plot]
                if audio_features and audio_features is not None:
                    # popping items here forces this model to be terminal (last model in the pipeline)
                    try:
                        features["audio_features"] = audio_features.popleft()  # list(audio_features)[-1]
                    except:
                        features["audio_features"] = None
        else:
            for result in previous_data:
                if isinstance(result, tuple) and result[0]:
                    # keep the frames object and extract the id from the info
                    # if keep_frames is None or result[keep_frames]
                    for f_idx, frame_data in enumerate(result[4]):
                        for plot_name, plot in result[1][f_idx].items():

                            if plot_name != "PLOT" and (inp_img_names_list is None or plot_name in inp_img_names_list):
                                #if f_idx in previous_maps:
                                #    previous_maps[f_idx].append(plot)
                                #else:
                                #    previous_maps[f_idx] = [plot]
                                if frame_data["frame_info"]["frame_id"] in previous_maps:
                                    previous_maps[frame_data["frame_info"]["frame_id"]].append(plot)
                                else:
                                    previous_maps[frame_data["frame_info"]["frame_id"]] = [plot]
        features["previous_maps"] = previous_maps
        features["available_keys"] = kwargs.get("available_keys", None)
        return features

    def annotate_frame(self, input_data, plotter,
                       show_det_target_map=True,
                       show_det_gaze_fov=False,
                       gaze_aggregation="sum",
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties = input_data

        properties = {**properties,
                      "show_det_gaze_fov": (show_det_gaze_fov, "toggle", (True, False)),
                      "show_det_target_map": (show_det_target_map, "toggle", (True, False))}

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": grouped_video_frames["PLOT"] + [["det_source_" + self.short_name,
                                                                         "det_transformed_" + self.short_name]],
                                "det_source_" + self.short_name: grouped_video_frames["captured"],
                                "det_transformed_" + self.short_name: grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}
        if grabbed_video:
            if not "target_maps" in info["frame_detections_" + self.short_name]:
                info["frame_detections_" + self.short_name]["target_maps"] = grouped_video_frames["captured"].copy()

            if gaze_aggregation == "amean":
                target_map = np.nanmean(info["frame_detections_" + self.short_name]["target_maps"][0], axis=0)
            elif gaze_aggregation == "gmean":
                target_map = gmean(info["frame_detections_" + self.short_name]["target_maps"][0], axis=0)
            elif gaze_aggregation == "sum":
                target_map = np.sum(info["frame_detections_" + self.short_name]["target_maps"][0], axis=0)
            else:
                target_map = info["frame_detections_" + self.short_name]["target_maps"][0][0]

            # find the peak
            peak_position = np.unravel_index(np.argmax(target_map), target_map.shape)
            # peak_value = target_map[peak_position]
            peak_position_norm = pixels_to_bounded_range(target_map.shape, peak_position, xy_bounds=(-1, 1))
            info["frame_detections_" + self.short_name]["SCREEN_gaze_target"] = peak_position_norm

            if show_det_target_map:
                target_map = target_map / np.max(np.abs(target_map))
                frame_transformed = plotter.plot_color_map(np.uint8(255 * target_map), color_map=color_map)
                if enable_transform_overlays:
                    frame_transformed = plotter.plot_alpha_overlay(grouped_video_frames["det_transformed_" + self.short_name],
                                                                   frame_transformed, alpha=0.4)
                else:
                    frame_transformed = plotter.resize(frame_transformed,
                                                       height=grouped_video_frames["det_transformed_" + self.short_name].shape[0],
                                                       width=grouped_video_frames["det_transformed_" + self.short_name].shape[1])
                grouped_video_frames["det_transformed_" + self.short_name] = frame_transformed
            if show_det_gaze_fov:
                frame_transformed = plotter.plot_fov_mask(target_map, (peak_position[1], peak_position[0]), radius=50)
                grouped_video_frames["det_transformed_" + self.short_name] = frame_transformed

            # clear target maps, we don't need them
            del info["frame_detections_" + self.short_name]["target_maps"]

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties
