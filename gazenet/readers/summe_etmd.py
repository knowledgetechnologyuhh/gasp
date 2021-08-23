"""
Class for reading and decoding the AVEyetracking[3] dataset composed of the ETMD [1] and SumMe [2] datasets

[1] Gygli, M., Grabner, H., Riemenschneider, H., & Van Gool, L. (2014).
    Creating summaries from user videos.
    In Proceedings of the European Conference on Computer Vision, 2014, pp. 505 - 520.

[2] Koutras, P., & Maragos, P. (2015).
    Perceptually based spatio-temporal computational framework for visual saliency estimation.
    Signal Processing: Image Commununication, 2015, pp. 15 - 31

[3] Tsiami, A., Koutras, P., Katsamanis, A., Vatakis, A., & Maragos, P. (2019).
    Signal Processing: Image Communication, 2019, pp. 186 - 200
"""

import os

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, extract_width_height_from_video, check_audio_in_video
from gazenet.utils.sample_processors import SampleReader, SampleProcessor


@ReaderRegistrar.register
class SumMeETMDSampleReader(SampleReader):
    def __init__(self, video_dir="datasets/aveyetracking/SumMe_ETMD/video",
                 audio_dir="datasets/aveyetracking/SumMe_ETMD/audio_mono",
                 annotations_file="datasets/aveyetracking/SumMe_ETMD/eyetracking/all_videos.mat",
                 audio_format="wav", video_format="mp4",
                 extract_thumbnails=True,
                 pickle_file="temp/summeetmd.pkl", mode=None, **kwargs):
        self.short_name = "summeetmd"
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.annotations_file = annotations_file
        self.audio_format = audio_format
        self.video_format = video_format
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        # single annotations file in matlab format
        annotations = sio.loadmat(self.annotations_file)
        # annotations['eye_data_all'][clip_name][0][0][1][2][3] -> [1]:x(0),y(1), [2]:video_frames_list, [3]:participants{10}

        for video_name in tqdm(sorted(os.listdir(self.video_dir)), desc="Samples Read"):
            if video_name.endswith("." + self.video_format):
                id = video_name.replace("." + self.video_format, "")
                audio_full_name = os.path.join(self.audio_dir, id + "." + self.audio_format)
                if os.path.isfile(audio_full_name):
                    has_audio = True
                else:
                    audio_full_name = os.path.join(self.video_dir, video_name)
                    has_audio = check_audio_in_video(os.path.join(self.video_dir, video_name))
                try:
                    # annotation assembly
                    annotation = annotations['eye_data_all'][id][0][0]

                    video_width_height = extract_width_height_from_video(os.path.join(self.video_dir, video_name))

                    annotation[0, :, :] *= video_width_height[0]
                    annotation[1, :, :] *= video_width_height[1]
                    annotation = annotation.astype(np.uint32)
                    self.samples.append({"id": id,
                                         "audio_name": audio_full_name,
                                         "video_name": os.path.join(self.video_dir, video_name),
                                         "video_width": video_width_height[0],
                                         "video_height": video_width_height[1],
                                         "video_thumbnail": extract_thumbnail_from_video(
                                             os.path.join(self.video_dir, video_name)) if self.extract_thumbnails else None,
                                         "len_frames": len(annotation[0]),
                                         "has_audio": has_audio,
                                         "annotation_name": id,
                                         "annotations": {"xyp": annotation}
                                         })
                    self.video_id_to_sample_idx[id] = len(self.samples) - 1
                    self.len_frames += self.samples[-1]["len_frames"]
                except:
                    print("Error: Access non-existent annotation " + id)

    @staticmethod
    def dataset_info():
        return {"summary": "TODO",
                "name": "SumMe (Koutras et al.) & ETMD (Gygli et al.) Datasets",
                "link": "TODO"}


@SampleRegistrar.register
class SumMeETMDSample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, SumMeETMDSampleReader)
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
                if participant is None:
                    fixation_participants = ann["xyp"][2, frame_index - 1, :]
                    fixation_annotations = np.vstack((ann["xyp"][0, frame_index - 1, :],
                                                      # empirically introduced a vertical shift of 20 pixels
                                                      ann["xyp"][1, frame_index - 1, :],
                                                      np.ones_like((ann["xyp"][0, frame_index - 1, :]))
                                                      # no fixation amplitude
                                                      )).transpose()
                else:
                    fixation_participants = ann["xyp"][2, frame_index - 1, participant]
                    fixation_annotations = np.vstack((ann["xyp"][0, frame_index - 1, participant],
                                                      ann["xyp"][1, frame_index - 1, participant],
                                                      np.ones_like((ann["xyp"][0, frame_index - 1, participant]))
                                                      # no fixation amplitude
                                                      )).transpose()

                info["frame_annotations"]["eye_fixation_participants"].append(fixation_participants)
                info["frame_annotations"]["eye_fixation_points"].append(fixation_annotations)
                if show_saliency_map:
                    video_frame_salmap = plotter.plot_fixations_density_map(video_frame_salmap, fixation_annotations,
                                                                            xy_std=(20, 20),
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
    reader = SumMeETMDSampleReader(mode="w")
