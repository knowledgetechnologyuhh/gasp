"""
Class for reading and decoding the Find who to look at [1] dataset


[1] Xu, M., Liu, Y., Hu, R., & He, F. (2018).
    Find who to look at: Turning from action to saliency.
    IEEE Transactions on Image Processing, 7(9), 4529-4544. IEEE.
"""

import os

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, check_audio_in_video, aggregate_frame_ranges
from gazenet.utils.sample_processors import SampleReader, SampleProcessor


@ReaderRegistrar.register
class FindWhoSampleReader(SampleReader):
    def __init__(self,
                 video_audio_dir="datasets/findwho",
                 annotations_file="datasets/findwho/video_database.mat",
                 database_type='video_database',
                 video_format="mp4",
                 audio_format="wav",
                 extract_thumbnails=True,
                 pickle_file="temp/findwho.pkl", mode=None, **kwargs):
        self.short_name = "findwho"
        self.video_dir = os.path.join(video_audio_dir, "raw_videos")
        self.audio_dir = os.path.join(video_audio_dir, "raw_audios")
        self.annotations_file = annotations_file
        self.database_type = database_type
        self.video_format = video_format
        self.audio_format = audio_format
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        # single annotations file in matlab format
        annotations = sio.loadmat(self.annotations_file)
        # annotations['video_data'][0][0][x] fixdata fields -> SubjectIndex,VideoIndex,Timestamp,Duration,x,y
        annotation_columns = [annotations[self.database_type]['fixdata_fields'][0][0][0][i].tolist()[0].replace(' ', '')
                              for i in range(len(annotations[self.database_type]['fixdata_fields'][0][0][0]))]
        annotation_fixations = pd.DataFrame(data=annotations[self.database_type]['fixdata'][0][0],
                                            columns=annotation_columns, index=None)
        video_names = [dI for dI in sorted(os.listdir(self.video_dir)) if dI.endswith("." + self.video_format)]
        for video_name in tqdm(video_names, desc="Samples Read"):
            sample_id = video_name.replace("." + self.video_format, "")
            try:
                # annotation assembly
                id_idx = int(sample_id) - 1
                video_annotations = annotation_fixations.loc[(annotation_fixations["VideoIndex"] == id_idx+1)]

                self.samples.append({"id": sample_id,
                                     "audio_name": os.path.join(self.audio_dir, sample_id + "." + self.audio_format),
                                     "video_name": os.path.join(self.video_dir, video_name),
                                     "video_fps": annotations[self.database_type]['videos_info'][0][0]['framerate_fps'][0][0][id_idx][0],
                                     "video_width": annotations[self.database_type]['videos_info'][0][0]['size'][0][0][id_idx][0],
                                     "video_height": annotations[self.database_type]['videos_info'][0][0]['size'][0][0][id_idx][1],
                                     "video_thumbnail": extract_thumbnail_from_video(
                                         os.path.join(self.video_dir, video_name)) if self.extract_thumbnails else None,
                                     "len_frames": annotations[self.database_type]['videos_info'][0][0]['frames'][0][0][id_idx][0],
                                     "has_audio": check_audio_in_video(os.path.join(self.video_dir, video_name)),
                                     "annotation_name": os.path.join(self.database_type, sample_id),
                                     "annotations": video_annotations,
                                     "participant_frames": {str(participant_id): aggregate_frame_ranges(video_annotations.loc[
                                         (video_annotations["SubjectIndex"] == participant_id)][["Timestamp", "GazeDuration"]].values.tolist())
                                                            for participant_id in video_annotations["SubjectIndex"].unique().tolist()}
                                     })
                self.video_id_to_sample_idx[sample_id] = len(self.samples) - 1
                self.len_frames += self.samples[-1]["len_frames"]
            except:
                print("Error: Access non-existent annotation " + sample_id)

    @staticmethod
    def dataset_info():
        return {"summary": "TODO",
                "name": "'Find Who to look at' Dataset (Xu et al.)",
                "link": "TODO"}


@SampleRegistrar.register
class FindWhoSample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, FindWhoSampleReader)
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
                    fps = self.reader.samples[self.index]["video_fps"]
                    ann = self.reader.samples[self.index]["annotations"]
                    if p_itr is not None:
                        ann = ann.loc[ann["SubjectIndex"] == p_itr]

                    fixation_annotations = ann.loc[(ann["Timestamp"] / 1000 <= frame_index / fps) &
                                                   (ann["Timestamp"] / 1000 + ann[
                                                       "GazeDuration"] / 1000 >= frame_index / fps)][
                        ["SubjectIndex", "FixationPointX", "FixationPointY", "GazeDuration"]]

                    fixation_participants = fixation_annotations.iloc[:, [0]].values
                    fixation_annotations = np.hstack((fixation_annotations.iloc[:, [1]].values,
                                                      fixation_annotations.iloc[:, [2]].values,
                                                      fixation_annotations.iloc[:, [3]].values))

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
    reader = FindWhoSampleReader(mode="w")
