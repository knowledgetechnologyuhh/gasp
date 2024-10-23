"""
Class for reading and decoding the DIEM [1] dataset

[1] Mital, P. K., Smith, T. J., Hill, R. L., & Henderson, J. M. (2011).
    Clustering of gaze during dynamic scene viewing is predicted by motion.
    Cognitive computation, 3(1), 5-24.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, extract_width_height_from_video
from gazenet.utils.sample_processors import SampleReader, SampleProcessor


@ReaderRegistrar.register
class DIEMSampleReader(SampleReader):
    def __init__(self, video_audio_annotations_dir="datasets/ave/diem",
                 video_format="mp4", audio_format="wav", annotation_format="txt", annotation_columns=None,
                 extract_thumbnails=True,
                 pickle_file="temp/diem.pkl", mode=None, **kwargs):
        self.short_name = "diem"
        self.video_audio_annotations_dir = video_audio_annotations_dir
        self.video_format = video_format
        self.audio_format = audio_format
        self.annotation_format = annotation_format
        self.annotation_columns = annotation_columns
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        if self.annotation_columns is None:
            self.annotation_columns = ["frame", "left_x", "left_y", "left_dil", "left_event", "right_x", "right_y",
                                  "right_dil", "right_event"]
        ids = [dI for dI in sorted(os.listdir(self.video_audio_annotations_dir)) if os.path.isdir(os.path.join(self.video_audio_annotations_dir, dI))]

        for sample_id in tqdm(ids, desc="Samples Read"):
            video_dir = os.path.join(self.video_audio_annotations_dir, sample_id, "video")
            video_name = sample_id + "." + self.video_format
            audio_dir = os.path.join(self.video_audio_annotations_dir, sample_id, "audio")
            audio_name = sample_id + "." + self.audio_format
            annotations_dir = os.path.join(self.video_audio_annotations_dir, sample_id, "event_data")
            annotations_name = "*" + sample_id + "." + self.annotation_format
            # [frame] [left_x] [left_y] [left_dil] [left_event] [right_x] [right_y] [right_dil] [right_event]
            # Frames are 30 video_frames_list per second, indexed at 1; x,y are screen coordinates; dil represents pupil dilation; and the event flag represents:
            # -1 = Error/dropped frame
            # 0 = Blink
            # 1 = Fixation
            # 2 = Saccade

            # read the annotations in all the files (participants?)
            annotations = []
            for part_id, annotation_path in enumerate(sorted(glob(os.path.join(annotations_dir, annotations_name)))):
                annotation_name = os.path.basename(annotation_path)
                part_name = annotation_name.replace("_" + sample_id + "." + self.annotation_format, "")
                # annotation assembly
                annotation = pd.read_csv(annotation_path, sep="\t", names=self.annotation_columns)
                annotation["participant"] = part_name
                annotation["participant_id"] = part_id
                annotations.append(annotation)
            video_width_height = extract_width_height_from_video(os.path.join(video_dir, video_name))
            self.samples.append({"id": sample_id,
                                 "audio_name": os.path.join(audio_dir, audio_name),
                                 "video_name": os.path.join(video_dir, video_name),
                                 "video_width": video_width_height[0],
                                 "video_height": video_width_height[1],
                                 "video_thumbnail": extract_thumbnail_from_video(
                                     os.path.join(video_dir, video_name)) if self.extract_thumbnails else None,
                                 "annotation_name": os.path.join(annotations_dir, annotations_name),
                                 "has_audio": True,
                                 "annotations": pd.concat(annotations, axis=0, ignore_index=True, sort=False)})
            self.video_id_to_sample_idx[sample_id] = len(self.samples) - 1
            self.samples[-1].update({"len_frames": int(self.samples[-1]["annotations"]["frame"].max())})
            self.len_frames += self.samples[-1]["len_frames"]

    @staticmethod
    def dataset_info():
        return {"summary": "TODO",
                "name": "DIEM Dataset (Mital et al.)",
                "link": "TODO"}


@SampleRegistrar.register
class DIEMSample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, DIEMSampleReader)
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

    def annotate_frame(self, input_data, plotter,
                       show_saliency_map=False,
                       show_fixation_locations=False,
                       eye="left",
                       participant=None,  # None means all participants will be plotted
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, _  = input_data

        properties = {"show_saliency_map": (show_saliency_map, "toggle", (True, False)),
                      "show_fixation_locations": (show_fixation_locations, "toggle", (True, False))}

        info = {**info, "frame_annotations": {
            "eye_fixation_points": [],
            "eye_fixation_participants": []
        }}
        # info["frame_info"]["dataset_name"] = self.reader.short_name
        # info["frame_info"]["video_id"] = self.reader.samples[self.index]["id"]
        # info["frame_info"]["frame_height"] = self.reader.samples[self.index]["video_height"]
        # info["frame_info"]["frame_width"] = self.reader.samples[self.index]["video_width"]

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": [["captured", "transformed_salmap", "transformed_fixmap"]],
                                "transformed_salmap": grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"]),
                                "transformed_fixmap": grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}

        try:
            frame_index = self.frame_index()
            video_frame_salmap = grouped_video_frames["transformed_salmap"]
            video_frame_fixmap = grouped_video_frames["transformed_fixmap"]

            if grabbed_video:
                ann = self.reader.samples[self.index]["annotations"]
                if participant is not None:
                    ann = ann.loc[ann["participant_id"] == participant]

                if eye == "left":
                    fixation_annotations = ann.loc[(ann["frame"] == frame_index) & (ann["left_event"] == 1)][
                        ["participant_id", "left_x", "left_y", "left_dil", "left_event"]]
                elif eye == "right":
                    fixation_annotations = ann.loc[(ann["frame"] == frame_index) & (ann["right_event"] == 1)][
                        ["participant_id", "right_x", "right_y", "right_dil"]]
                else:
                    raise NotImplementedError("Implement both eyes at once")

                fixation_participants = fixation_annotations.iloc[:, [0]].values
                fixation_annotations = np.hstack((fixation_annotations.iloc[:, [1]].values,
                                                  # empirically introduced a vertical shift of 100 pixels
                                                  fixation_annotations.iloc[:, [2]].values - 100,
                                                  fixation_annotations.iloc[:, [3]].values))

                # fixation_annotations = np.squeeze(fixation_annotations, axis=-1)
                info["frame_annotations"]["eye_fixation_participants"].append(fixation_participants)
                info["frame_annotations"]["eye_fixation_points"].append(fixation_annotations)
                if fixation_annotations.shape[0] != 0:
                    if show_saliency_map:
                        video_frame_salmap = plotter.plot_fixations_density_map(video_frame_salmap, fixation_annotations,
                                                                                xy_std=(60, 60),
                                                                                color_map=color_map,
                                                                                alpha=0.4 if enable_transform_overlays else 1.0)
                    if show_fixation_locations:
                        video_frame_fixmap = plotter.plot_fixations_locations(video_frame_fixmap, fixation_annotations,
                                                                              radius=1)

            grouped_video_frames["transformed_salmap"] = video_frame_salmap
            grouped_video_frames["transformed_fixmap"] = video_frame_fixmap
        except:
            pass

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def get_participant_frame_range(self, participant_id):
        raise NotImplementedError


if __name__ == "__main__":
    reader = DIEMSampleReader(mode="w")
