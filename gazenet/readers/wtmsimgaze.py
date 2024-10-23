"""
Class for reading and decoding the WTM simulated gaze videos [1] dataset used in the cognitive robotic simulation [2] study.

[1] Kerzel, M., & Wermter, S. (2020).
    Towards a Data Generation Framework for Affective Shared Perception and Social Cue Learning Using Virtual Avatars.
    In Workshop on Affective Shared Perception.

[2] Fu, D., Abawi, F., Carneiro, H., Kerzel, M., Chen, Z., Strahl, E., Liu, X., & Wermter, S. (2020).
    A Trained Humanoid Robot can Perform Human-Like Crossmodal Social Attention and Conflict Resolution.
    In International Journal of Social Robotics, 2023, 15, pp. 1325-1341.
"""

import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, check_audio_in_video, extract_len_from_video, extract_width_height_from_video
from gazenet.utils.sample_processors import SampleReader, SampleProcessor

FLIP_DIRECTION_LOOKUP = {"left": "right", "right": "left", "normal": "normal"}
FLIP_CONGRUENCY_LOOKUP = {"congruent": "incongruent", "incongruent": "congruent", "neutral": "neutral"}


@ReaderRegistrar.register
class WTMSimGazeSampleReader(SampleReader):
    def __init__(self, video_dir="datasets/wtmsimgaze2020/videos",
                 annotations_file="datasets/wtmsimgaze2020/video_conditions.csv",
                 video_format="mp4", annotation_columns=None,
                 extract_thumbnails=True,
                 pickle_file="temp/wtmsimgaze.pkl", mode=None, **kwargs):
        self.short_name = "wtmsimgaze"
        self.video_dir = video_dir
        self.annotations_file = annotations_file
        self.video_format = video_format
        self.annotation_columns = annotation_columns
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        if self.annotation_columns is None:
            self.annotation_columns = ["video_id", "auditory_localization", "visualcue_localization",
                                       "conditions_congruency", "fix_duration_1", "fix_duration_2"]

        # single metadata (annotation) file in csv format
        annotations = pd.read_csv(self.annotations_file, names=self.annotation_columns, header=0)

        for video_name in tqdm(sorted(os.listdir(self.video_dir)), desc="Samples Read"):
            if video_name.endswith("." + self.video_format):
                sample_id = video_name.replace("." + self.video_format, "")

                try:
                    # annotation assembly
                    annotation = annotations[annotations["video_id"] == sample_id]
                    len_frames, _, fps = extract_len_from_video(os.path.join(self.video_dir, video_name))
                    video_width_height = extract_width_height_from_video(os.path.join(self.video_dir, video_name))
                    self.samples.append({"id": sample_id,
                                         "audio_name": os.path.join(self.video_dir, video_name),
                                         "video_name": os.path.join(self.video_dir, video_name),
                                         "video_fps": fps,
                                         "video_width": video_width_height[0],
                                         "video_height": video_width_height[1],
                                         "video_thumbnail": extract_thumbnail_from_video(os.path.join(self.video_dir, video_name)) if self.extract_thumbnails else None,
                                         "len_frames": len_frames,
                                         "has_audio": check_audio_in_video(os.path.join(self.video_dir, video_name)),
                                         "annotation_name": sample_id,
                                         "annotations": {"metadata": annotation}
                                         })
                    self.video_id_to_sample_idx[sample_id] = len(self.samples) - 1
                    self.len_frames += self.samples[-1]["len_frames"]
                except:
                    print("Error: Access non-existent annotation " + sample_id)

    @staticmethod
    def dataset_info():
        return {"summary": "TODO",
                "name": "WTM Simulated Gaze 2020 (Kerzel and Wermter; Fu, Abawi, et al.)",
                "link": "https://osf.io/fbncu/"}


@SampleRegistrar.register
class WTMSimGazeSample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, WTMSimGazeSampleReader)
        self.short_name = reader.short_name
        self.reader = reader
        self.index = index

        if frame_index > 0:
            self.goto_frame(frame_index)
        super().__init__(width=width, height=height, **kwargs)
        next(self)

    def __next__(self):
        with self.read_lock:
            self.index += 1
            self.index %= len(self.reader.samples)
        curr_metadata = self.reader.samples[self.index]
        self.load(curr_metadata)
        return curr_metadata

    def __len__(self):
        return len(self.reader)

    def next(self):
        return next(self)

    def goto(self, name, by_index=True):
        if by_index:
            index = name
        else:
            index = self.reader.video_id_to_sample_idx[name]

        with self.read_lock:
            self.index = index
        curr_metadata = self.reader.samples[self.index]
        self.load(curr_metadata)
        return curr_metadata

    # TODO (fabawi): flip audio not implemented yet
    def annotate_frame(self, input_data, plotter,
                       flip_video=False,
                       flip_audio=False,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, _ = input_data

        properties = {"flip_video": (flip_video, "toggle", (True, False)),
                      "flip_audio": (flip_audio, "toggle", (True, False))}

        info = {**info, "frame_annotations": {
            "auditory_localization": None,
            "visualcue_localization": None,
            "conditions_congruency": None
        }}
        # info["frame_info"]["dataset_name"] = self.reader.short_name
        # info["frame_info"]["video_id"] = self.reader.samples[self.index]["id"]
        # info["frame_info"]["frame_height"] = self.reader.samples[self.index]["video_height"]
        # info["frame_info"]["frame_width"] = self.reader.samples[self.index]["video_width"]

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": [["captured", "transformed_captured"]],
                                "transformed_captured": grouped_video_frames["captured"]}

        try:
            if grabbed_video:
                ann = self.reader.samples[self.index]["annotations"]

                if flip_audio:
                    info["frame_annotations"]["auditory_localization"] = \
                        FLIP_DIRECTION_LOOKUP[ann["metadata"]["auditory_localization"]]
                    info["frame_annotations"]["conditions_congruency"] = \
                        FLIP_CONGRUENCY_LOOKUP[ann["metadata"]["conditions_congruency"]]
                else:
                    info["frame_annotations"]["auditory_localization"] = ann["metadata"]["auditory_localization"]
                    info["frame_annotations"]["conditions_congruency"] = ann["metadata"]["conditions_congruency"]

                if flip_video:
                    info["frame_annotations"]["visualcue_localization"] = \
                        FLIP_DIRECTION_LOOKUP[ann["metadata"]["visualcue_localization"]]
                    info["frame_annotations"]["conditions_congruency"] = \
                        FLIP_CONGRUENCY_LOOKUP[info["frame_annotations"]["conditions_congruency"]]
                    grouped_video_frames["transformed_captured"] = plotter.flip_horizontal(grouped_video_frames["transformed_captured"])
                else:
                    info["frame_annotations"]["visualcue_localization"] = ann["metadata"]["visualcue_localization"]
                    # info["frame_annotations"]["conditions_congruency"] = info["frame_annotations"]["conditions_congruency"]



        except:
            pass

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def get_participant_frame_range(self, participant_id):
        raise NotImplementedError


if __name__ == "__main__":
    reader = WTMSimGazeSampleReader(mode="w")
