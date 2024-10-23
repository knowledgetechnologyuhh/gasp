"""
Class for reading and decoding the Coutrot1 [1] and Coutrot2 [2] datasets


[1] Coutrot, A., & Guyader, N. (2014).
    How saliency, faces, and sound influence gaze in dynamic social scenes.
    Journal of vision, 14(8), 5-5.

[2] Coutrot, A., & Guyader, N. (2015).
    An efficient audiovisual saliency model to infer eye positions when looking at conversations.
    In 2015 23rd European Signal Processing Conference (EUSIPCO) (pp. 1531-1535). IEEE.
"""

import os

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_thumbnail_from_video, check_audio_in_video
from gazenet.utils.sample_processors import SampleReader, SampleProcessor


@ReaderRegistrar.register
class CoutrotSampleReader(SampleReader):
    def __init__(self, video_dir, annotations_file,
                 database_type, # database_type = 'Coutrot_Database1'| 'Coutrot_Database2'
                 auditory_condition, video_format="avi",
                 extract_thumbnails=True,
                 pickle_file=None, mode=None, **kwargs):
        self.short_name = "coutrot"
        self.video_dir = video_dir
        self.annotations_file = annotations_file
        self.database_type = database_type
        self.auditory_condition = auditory_condition
        self.video_format = video_format
        self.extract_thumbnails = extract_thumbnails

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        # single annotations file in matlab format
        annotations = sio.loadmat(self.annotations_file)
        # annotations['Coutrot_Database1'][0][0][x] Auditory condition -> clips in red on webpage are actually excluded
        # annotations['Coutrot_Database1']['OriginalSounds'][0][0]['clip_1'][0][0][0][0]['data'][1][2][3] -> [1]:x(0),y(1),[2]: video_frames_list, [3]: participants?

        for video_name in tqdm(sorted(os.listdir(self.video_dir)), desc="Samples Read"):
            if video_name.endswith("." + self.video_format):
                sample_id = video_name.replace("." + self.video_format, "")

                try:
                    # annotation assembly
                    annotation = annotations[self.database_type][self.auditory_condition][0][0][sample_id][0][0][0]
                    self.samples.append({"id": sample_id,
                                         "audio_name": os.path.join(self.video_dir, video_name),
                                         "video_name": os.path.join(self.video_dir, video_name),
                                         "video_fps": annotation['info'][0]['fps'][0][0][0][0],
                                         "video_width": annotation['info'][0]['vidwidth'][0][0][0][0],
                                         "video_height": annotation['info'][0]['vidheight'][0][0][0][0],
                                         "video_thumbnail": extract_thumbnail_from_video(
                                             os.path.join(self.video_dir, video_name)) if self.extract_thumbnails else None,
                                         "len_frames": annotation['info'][0]['nframe'][0][0][0][0],
                                         "has_audio": check_audio_in_video(os.path.join(self.video_dir, video_name)),
                                         "annotation_name": os.path.join(self.database_type, self.auditory_condition, sample_id),
                                         "annotations": {"xyp": annotation['data'][0]}
                                         })
                    self.video_id_to_sample_idx[sample_id] = len(self.samples) - 1
                    self.len_frames += self.samples[-1]["len_frames"]
                except:
                    print("Error: Access non-existent annotation " + sample_id)

    @staticmethod
    def dataset_info():
        return {"summary": "NONE",
                "name": "Coutrot Dataset",
                "link": "http://antoinecoutrot.magix.net/public/databases.html"}

@ReaderRegistrar.register
class Coutrot1SampleReader(CoutrotSampleReader):
    def __init__(self, video_dir="datasets/ave/database1/ERB3_Stimuli",
                 annotations_file="datasets/ave/database1/coutrot_database1.mat",
                 database_type='Coutrot_Database1', auditory_condition='OriginalSounds',
                 pickle_file="temp/coutrot1.pkl", mode=None, **kwargs):
        super().__init__(video_dir=video_dir, annotations_file=annotations_file,
                         database_type=database_type, auditory_condition=auditory_condition,
                         pickle_file=pickle_file, mode=mode, **kwargs)
        self.short_name = "coutrot1"

    @staticmethod
    def dataset_info():
        return {"summary": "Coutrot 1 dataset consists of eye tracking data for 18 participants on 60 videos, with clips 46-60 retained only (faces).",
                "name": "Coutrot Dataset1 (Coutrot et al.)",
                "link": "http://antoinecoutrot.magix.net/public/databases.html"}

@ReaderRegistrar.register
class Coutrot2SampleReader(CoutrotSampleReader):
    def __init__(self, video_dir="datasets/ave/database2/ERB4_Stimuli",
                 annotations_file="datasets/ave/database2/coutrot_database2.mat",
                 database_type='Coutrot_Database2', auditory_condition='AudioVisual',
                 pickle_file="temp/coutrot2.pkl", mode=None, **kwargs):
        super().__init__(video_dir=video_dir, annotations_file=annotations_file,
                         database_type=database_type, auditory_condition=auditory_condition,
                         pickle_file=pickle_file, mode=mode, **kwargs)
        self.short_name = "coutrot2"

    @staticmethod
    def dataset_info():
        return {"summary": "Coutrot 2 dataset consists of eye tracking data for 20 participants on 15 videos, each with 4 persons having a meeting.",
                "name": "Coutrot Dataset2 (Coutrot and Guyader)",
                "link": "http://antoinecoutrot.magix.net/public/databases.html"}

@SampleRegistrar.register
class CoutrotSample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, CoutrotSampleReader)
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
                                                      ann["xyp"][1, frame_index - 1, :] - 20,
                                                      np.ones_like((ann["xyp"][0, frame_index - 1, :]))
                                                      # no fixation amplitude
                                                      )).transpose()
                else:
                    fixation_participants = ann["xyp"][2, frame_index - 1, participant]
                    fixation_annotations = np.vstack((ann["xyp"][0, frame_index - 1, participant],
                                                      # empirically introduced a vertical shift of 20 pixels
                                                      ann["xyp"][1, frame_index - 1, participant] - 20,
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
                    video_frame_fixmap = plotter.plot_fixations_locations(video_frame_fixmap, fixation_annotations, radius=1)

            grouped_video_frames["transformed_salmap"] = video_frame_salmap
            grouped_video_frames["transformed_fixmap"] = video_frame_fixmap

        except:
            pass

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def get_participant_frame_range(self, participant_id):
        raise NotImplementedError


if __name__ == "__main__":
    reader1 = Coutrot1SampleReader(mode="w")
    reader2 = Coutrot2SampleReader(mode="w")
