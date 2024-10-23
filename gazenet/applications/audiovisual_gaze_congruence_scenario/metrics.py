import random
import math
from collections import deque

import numpy as np
import pandas as pd
import cv2

from gazenet.utils.registrar import *


@MetricsRegistrar.register
class HRIAudioVisualCongruenceMetrics(object):
    def __init__(self, save_file="logs/metrics/hriavcongruence.csv", dataset_name="", video_name="",
                 metrics_list=["pred_location_x", "pred_location_y", "pred_direction", "pred_accuracy"], *args, **kwargs):

        self.save_file = save_file
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        self.metrics_list = metrics_list
        self._dataset_name = dataset_name
        self._video_name = video_name  # this is the group name, but kept as such for cosistency with other metrics

        self.columns = ["participant_idx", "video_group_id", "dataset", "frame_id",
                        "auditory_localization", "visualcue_localization", "condition_congruency"] + metrics_list

        # scores contain the clean (complete results)
        if os.path.exists(save_file):
            self.scores = pd.read_csv(save_file, header=0)
        else:
            self.scores = pd.DataFrame(columns=self.columns)
        # accumulator contains the intermediate outputs of each needed model
        self.accumulator = pd.DataFrame(columns=self.columns)

    def set_new_name(self, vid_name):
        collated_metrics = self.accumulate_metrics()
        self._video_name = vid_name
        return collated_metrics

    def save(self):
        self.scores.to_csv(self.save_file,  index=False)

    def accumulate_metrics(self, intermed_save=True):
        # filter metrics
        filtered_metrics = self.accumulator.dropna(thresh=None)

        # drop all accumulator samples that don't belong to current video group and are still incomplete
        self.accumulator = self.accumulator[(self.accumulator["video_group_id"] == self._video_name) &
                                            self.accumulator.isnull().any(axis=1)]

        if not filtered_metrics.empty:
            self.scores = self.scores.append(filtered_metrics, ignore_index=True)

        # save to file after every video
        if intermed_save:
            self.save()
        return filtered_metrics.to_dict('r')

    def add_metrics(self, returns, models, *args, **kwargs):
        # three stages:
        # 1) get the returns of the video player
        # 2) associate 1st perception with video player
        # 3) associate frame_id from greedy gaze with perception frame_id

        metrics_args = {}
        for idx_model, model_data in enumerate(models):
            idx_return = idx_model + 2 if isinstance(returns[0], dict) else idx_model
            # 1) get the returns of the video player
            if models[idx_model][0] == "PlayHRIVideoApplication":
                vid_properties = pd.DataFrame(returns[idx_return][4]["frame_annotations"], index=[0])
                vid_properties["video_group_id"] = self._video_name
                vid_properties["dataset"] = self._dataset_name
                self.accumulator = pd.concat([self.accumulator, vid_properties], ignore_index=True)

            # 3) associate frame_id from greedy gaze with perception frame_id
            if "GreedyGazeInference" in models[idx_model][0] and returns[0][0]:
                screen_gaze_target = returns[idx_return][4][0]["frame_detections_greedygaze"]["SCREEN_gaze_target"]
                metrics = self.compute_metrics(gt_info=self.accumulator[self.accumulator["frame_id"] == returns[idx_return][4][0]['frame_info']['frame_id']],
                                               pred_screen_target=screen_gaze_target,
                                               **metrics_args)
                self.accumulator[self.accumulator["frame_id"] == returns[idx_return][4][0]['frame_info']['frame_id']] = metrics

        # 2) get the returns of the video player
        if isinstance(returns[0], dict):
            video_properties = self.accumulator.iloc[-1, :]
            if video_properties.notna()["frame_id"]:
                raise AssertionError("There are no video playbacks with unacquired frame_id preceding the input "
                                     "acquisition. Something wrong in the HRI experiment pipeline pipeline")
            video_properties = pd.concat([video_properties] * len(returns[0]["info_list"]), axis=1).transpose()
            self.accumulator = self.accumulator[:-1]
            for frame_idx, frame in enumerate(returns[0]["info_list"]):
                video_properties.iloc[frame_idx, video_properties.columns.get_loc("frame_id")] = frame["frame_info"]["frame_id"]
            self.accumulator = pd.concat([self.accumulator, video_properties], ignore_index=True)

        return None

    def compute_metrics(self, gt_info, pred_screen_target, **kwargs):
        if "pred_location_x" in self.metrics_list:
            gt_info["pred_location_x"] = pred_screen_target[1]
        if "pred_location_y" in self.metrics_list:
            gt_info["pred_location_y"] = pred_screen_target[0]
        if "pred_direction" in self.metrics_list:
            gt_info["pred_direction"] = "left" if pred_screen_target[1] > 0 else "right"
        if "pred_accuracy" in self.metrics_list:
            gt_info["pred_accuracy"] = np.where((gt_info["auditory_localization"] == gt_info["pred_direction"]), 1, 0)
        return gt_info


