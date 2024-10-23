"""
Class for reading and decoding the Find who to look at [1] dataset

[1] Liu, Y., Qiao, M., Xu, M., Li, B., Hu, W., & Borji, A. (2020).
    Learning to Predict Salient Faces: A Novel Visual-Audio Saliency Model.
    In European Conference on Computer Vision (ECCV).
"""

import os

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, check_audio_in_video, extract_len_from_video, extract_width_height_from_video
from gazenet.utils.sample_processors import SampleReader, SampleProcessor


@ReaderRegistrar.register
class MVVASampleReader(SampleReader):
    def __init__(self,
                 video_audio_dir="datasets/mvva/our_database_video",
                 annotation_dir="datasets/mvva/all_videos_all_frames_fixation",
                 video_format="mp4",
                 audio_format="wav",
                 annotation_format="npy",
                 extract_thumbnails=True,
                 pickle_file="temp/mvva.pkl", mode=None, **kwargs):
        self.short_name = "mvva"
        self.video_dir = os.path.join(video_audio_dir, "raw_videos")
        self.audio_dir = os.path.join(video_audio_dir, "raw_audios")
        self.annotation_dir = annotation_dir
        self.video_format = video_format
        self.audio_format = audio_format
        self.annotation_format = annotation_format
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        video_names = [dI for dI in sorted(os.listdir(self.video_dir)) if dI.endswith("." + self.video_format)]
        for video_name in tqdm(video_names, desc="Samples Read"):
            sample_id = video_name.replace("." + self.video_format, "")
            try:
                # annotation assembly
                video_annotations = np.load(os.path.join(self.annotation_dir, sample_id + "." + self.annotation_format))

                len_frames, _, fps = extract_len_from_video(os.path.join(self.video_dir, video_name))
                video_width_height = extract_width_height_from_video(os.path.join(self.video_dir, video_name))

                self.samples.append({"id": sample_id,
                                     "audio_name": os.path.join(self.audio_dir, sample_id + "." + self.audio_format),
                                     "video_name": os.path.join(self.video_dir, video_name),
                                     "video_fps": fps,
                                     "video_width": video_width_height[0],
                                     "video_height": video_width_height[1],
                                     "video_thumbnail": extract_thumbnail_from_video(
                                         os.path.join(self.video_dir, video_name)) if self.extract_thumbnails else None,
                                     "len_frames": len_frames,
                                     "has_audio": check_audio_in_video(os.path.join(self.video_dir, video_name)),
                                     "annotation_name": os.path.join(self.annotation_dir, sample_id + "." + self.annotation_format),
                                     "annotations": {"xyp": video_annotations}
                                     })
                self.video_id_to_sample_idx[sample_id] = len(self.samples) - 1
                self.len_frames += self.samples[-1]["len_frames"]
            except:
                print("Error: Access non-existent annotation " + sample_id)

    @staticmethod
    def dataset_info():
        return {"summary": "TODO",
                "name": "MVVA Dataset (Liu et al.)",
                "link": "TODO"}


@SampleRegistrar.register
class MVVASample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, MVVASampleReader)
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
                       participant=None,  # None means all participants will be plotted
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, _ = input_data

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

        if not isinstance(participant, list):  # if participant list is provided, plot each saliency map separately
            grouped_video_frames = {**grouped_video_frames,
                                    "PLOT": [["captured", "transformed_salmap", "transformed_fixmap"]],
                                    "transformed_salmap": grouped_video_frames["captured"]
                                    if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"]),
                                    "transformed_fixmap": grouped_video_frames["captured"]
                                    if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}
            participant = [participant]
        else:
            grouped_video_frames = {**grouped_video_frames,
                                    "PLOT": [["captured",
                                              "transformed_salmap_p" + str(participant[0]),
                                              "transformed_fixmap_p" + str(participant[0])]]}

        for p_itr in participant:
            try:
                frame_index = self.frame_index()
                if len(participant) > 1:
                    grouped_video_frames["transformed_salmap_p" + str(p_itr)] = grouped_video_frames["captured"] \
                        if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])
                    grouped_video_frames["transformed_fixmap_p" + str(p_itr)] = grouped_video_frames["captured"] \
                        if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])
                    video_frame_salmap = grouped_video_frames["transformed_salmap_p" + str(p_itr)]
                    video_frame_fixmap = grouped_video_frames["transformed_fixmap_p" + str(p_itr)]
                else:
                    video_frame_salmap = grouped_video_frames["transformed_salmap"]
                    video_frame_fixmap = grouped_video_frames["transformed_fixmap"]

                if grabbed_video:
                    ann = self.reader.samples[self.index]["annotations"]
                    if p_itr is None:
                        fixation_participants = range(1, ann["xyp"][frame_index - 1, :].shape[-1] + 1)
                        fixation_annotations = np.vstack((ann["xyp"][frame_index - 1, 0][:],
                                                          ann["xyp"][frame_index - 1, 1][:],
                                                          np.ones_like((ann["xyp"][frame_index - 1, :][0]))
                                                          # no fixation amplitude
                                                          )).transpose()
                    else:
                        fixation_participants = p_itr
                        fixation_annotations = np.vstack((ann["xyp"][frame_index - 1, 0][p_itr - 1],
                                                          ann["xyp"][frame_index - 1, 1][p_itr - 1],
                                                          np.ones_like((ann["xyp"][frame_index - 1, 0][0]))
                                                          # no fixation amplitude
                                                          )).transpose()

                    # fixation_annotations = np.squeeze(fixation_annotations, axis=-1)
                    info["frame_annotations"]["eye_fixation_participants"].append(fixation_participants)
                    info["frame_annotations"]["eye_fixation_points"].append(fixation_annotations)
                    if fixation_annotations.shape[0] != 0:
                        if show_saliency_map:
                            video_frame_salmap = plotter.plot_fixations_density_map(video_frame_salmap,
                                                                                    fixation_annotations,
                                                                                    xy_std=(66, 66),
                                                                                    color_map=color_map,
                                                                                    alpha=0.4 if enable_transform_overlays else 1.0)
                        if show_fixation_locations:
                            video_frame_fixmap = plotter.plot_fixations_locations(video_frame_fixmap,
                                                                                  fixation_annotations,
                                                                                  radius=1)
                if len(participant) > 1:
                    grouped_video_frames["transformed_salmap_p" + str(p_itr)] = video_frame_salmap
                    grouped_video_frames["transformed_fixmap_p" + str(p_itr)] = video_frame_fixmap
                else:
                    grouped_video_frames["transformed_salmap"] = video_frame_salmap
                    grouped_video_frames["transformed_fixmap"] = video_frame_fixmap
            except:
                pass

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def get_participant_frame_range(self, participant_id):
        raise NotImplementedError


if __name__ == "__main__":
    reader = MVVASampleReader(mode="w")
