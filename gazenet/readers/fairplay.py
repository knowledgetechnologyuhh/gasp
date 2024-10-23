"""
Class for reading and decoding the FAIR-Play [1] dataset

[1] Gao, R., & Grauman, K. (2019).
    2.5D Visual Sound.
    In Conference on Computer Vision and Pattern Recognition (CVPR).
"""

import os
from glob import glob
import numpy as np
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, extract_width_height_from_video, extract_len_from_video, open_image
from gazenet.utils.sample_processors import SampleReader, SampleProcessor


@ReaderRegistrar.register
class FAIRPlaySampleReader(SampleReader):
    def __init__(self, video_dir="datasets/fairplay/videos",
                 audio_dir="datasets/fairplay/binaural_audios",
                 annotation_dir="datasets/fairplay/maps",
                 video_format="mp4", audio_format="wav", annotation_format="png",
                 extract_thumbnails=True,
                 pickle_file="temp/fairplay.pkl", mode=None, **kwargs):
        self.short_name = "fairplay"
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.video_format = video_format
        self.audio_format = audio_format
        self.annotation_format = annotation_format
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        ids = [dI.replace("." + self.annotation_format, "") for dI in sorted(os.listdir(self.annotation_dir))
               if os.path.isfile(os.path.join(self.annotation_dir, dI)) and dI.endswith(self.annotation_format)]

        for sample_id in tqdm(ids, desc="Samples Read"):
            try:
                id_stripped = sample_id.split("_")[0]
                video_name = id_stripped + "." + self.video_format
                audio_name = id_stripped + "." + self.audio_format
                annotation_name = sample_id + "." + self.annotation_format

                # read the annotation images. Keep as list to maintain consistancy
                annotation = {"audiomap": open_image(os.path.join(self.annotation_dir, annotation_name), gray=True)}

                video_width_height = extract_width_height_from_video(os.path.join(self.video_dir, video_name))
                self.samples.append({"id": id_stripped,
                                     "audio_name": os.path.join(self.audio_dir, audio_name),
                                     "video_name": os.path.join(self.video_dir, video_name),
                                     "video_width": video_width_height[0],
                                     "video_height": video_width_height[1],
                                     "video_thumbnail": extract_thumbnail_from_video(
                                         os.path.join(self.video_dir, video_name)) if self.extract_thumbnails else None,
                                     "len_frames": extract_len_from_video(os.path.join(self.video_dir, video_name))[0],
                                     "annotation_name": os.path.join(self.annotation_dir, annotation_name),
                                     "has_audio": True,
                                     "annotations": annotation})
                self.video_id_to_sample_idx[id_stripped] = len(self.samples) - 1
                self.len_frames += self.samples[-1]["len_frames"]
            except:
                print("Error: Access non-existent annotation " + sample_id)

    @staticmethod
    def dataset_info():
        return {"summary": "TODO",
                "name": "FAIR-Play Dataset (Gao and Grauman)",
                "link": "TODO"}


@SampleRegistrar.register
class FAIRPlaySample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, FAIRPlaySampleReader)
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
                       show_audio_map=True,
                       enable_transform_overlays=True,
                       color_map=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, _ = input_data

        properties = {"show_audio_map": (show_audio_map, "toggle", (True, False))}

        info = {**info, "frame_annotations": {}}
        # info["frame_info"]["dataset_name"] = self.reader.short_name
        # info["frame_info"]["video_id"] = self.reader.samples[self.index]["id"]
        # info["frame_info"]["frame_height"] = self.reader.samples[self.index]["video_height"]
        # info["frame_info"]["frame_width"] = self.reader.samples[self.index]["video_width"]

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": [["captured", "transformed_audiomap"]],
                                "transformed_audiomap": grouped_video_frames["captured"]
                                if enable_transform_overlays else np.zeros_like(grouped_video_frames["captured"])}
        try:
            video_frame_audiomap = grouped_video_frames["transformed_audiomap"]

            if grabbed_video:
                ann = self.reader.samples[self.index]["annotations"]
                audiomap_annotations = plotter.plot_color_map(ann["audiomap"], color_map=color_map)

                if show_audio_map:
                    video_frame_audiomap = plotter.plot_alpha_overlay(video_frame_audiomap,
                                                                      audiomap_annotations,
                                                                      xy_min=None, xy_max=None,
                                                                      alpha=0.4 if enable_transform_overlays else 1.0,
                                                                      interpolation="nearest")

            grouped_video_frames["transformed_audiomap"] = video_frame_audiomap
        except:
            pass

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def get_participant_frame_range(self, participant_id):
        raise NotImplementedError


if __name__ == "__main__":
    reader = FAIRPlaySampleReader(mode="w")
