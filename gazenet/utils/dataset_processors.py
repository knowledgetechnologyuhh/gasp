import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from gazenet.utils.registrar import *
from gazenet.utils.helpers import extract_width_height_thumbnail_from_image
from gazenet.utils.sample_processors import SampleReader, SampleProcessor

try:  # pickle with protocol 5 if python<3.8
    import pickle5 as pickle
except:
    import pickle


class BaseDataset(Dataset):
    def __init__(self, csv_file, video_dir, annotation_dir, classes,
                 inp_img_names_list, inp_img_width=320, inp_img_height=240, inp_img_transform=None,
                 orig_img_width=1280, orig_img_height=720,
                 sequence_len=1, sample_stride=1, sample_overlap=False, replicate_class_samples=True):
        """
        Args:
            csv_file (string): Path to the csv file with the sample descriptions
            video_dir (string): Directory with all the frames separated into folders
            annotation_dir (string): Directory with all the annotations separated into files
            inp_img_names_list (list): List of the input image prefixes
            inp_img_transform (callable, optional): Optional transform applied to the input image
            sequence_len (int): Length of an image sequence. If None, the sequence dimension is removed
            sample_stride (int): Number of frames skipped between timesteps
            sample_overlap (bool): Traverse every timestep. If False, scanned samples will be skipped in following itrs
            replicate_class_samples (bool): replicates each participant point as new sample rather than returning all
        """
        self.samples = pd.read_csv(csv_file, dtype={"video_id": "string"})
        self.samples = self.samples[(self.samples["scene_type"] == "Social")].reset_index()
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.input_img_width = inp_img_width
        self.input_img_height = inp_img_height
        self.input_img_names_list = inp_img_names_list
        self.input_img_transform = inp_img_transform
        self.sequence_len = sequence_len
        self.sample_stride = sample_stride
        self.sample_overlap = sample_overlap
        self.replicate_class_samples = replicate_class_samples

        self.img_factor = (inp_img_width / orig_img_width, inp_img_height / orig_img_height)

        self.enumerate_video_frames()

    def enumerate_video_frames(self):
        self.samples["frames_len"] = self.samples.apply(lambda row:
                                                        len(os.listdir(os.path.join(self.annotation_dir if
                                                                                    self.annotation_dir else
                                                                                    self.video_dir, row["dataset"],
                                                                                    row["video_id"]))), axis=1)
        if self.classes and self.replicate_class_samples:
            old_len = len(self.samples)
            self.samples = pd.concat([self.samples]*len(self.classes))
            self.samples = self.samples.reset_index(drop=True)
            self.samples['participant_id'] = self.samples.index // old_len + 1

        self.samples["frame_pointer"] = self.samples["frames_len"].rolling(min_periods=1,
                                                                           window=len(self.samples)).sum()

    def __len__(self):
        if self.sample_overlap:
            return int(self.samples["frame_pointer"].iloc[-1])
        else:
            return int(self.samples["frame_pointer"].iloc[-1] // self.sample_stride)

    def __getfilenames__(self, global_idx):
        if not self.sample_overlap:
            global_idx *= self.sample_stride

        idx = self.samples.index[self.samples["frame_pointer"] > global_idx].tolist()[0]

        ds_name = self.samples.loc[idx, "dataset"]
        vid_name = self.samples.loc[idx, "video_id"]
        vid_len = self.samples.loc[idx, "frames_len"]

        if self.replicate_class_samples:
            cat_id = self.samples.loc[idx, "participant_id"]
            cat_name = "p" + str(cat_id)
        elif self.classes:
            cat_id = random.randrange(1, len(self.classes)+1)
            cat_name = "p" + str(cat_id)
        else:
            cat_id = 0
            cat_name = None

        curr_idx = int(global_idx - self.samples.loc[idx, "frame_pointer"] + self.samples.loc[idx, "frames_len"] + 1)

        if self.annotation_dir:
            vid_anno_path = os.path.join(self.annotation_dir, ds_name, vid_name)
            annotations_paths = [os.path.join(vid_anno_path, str(min(vid_len, frame_idx))) for frame_idx in
                                range(curr_idx, curr_idx + (self.sequence_len * self.sample_stride), self.sample_stride)]
        else:
            annotations_paths = None
        if self.video_dir:
            vid_img_path = os.path.join(self.video_dir, ds_name, vid_name)
            imgs_paths = [os.path.join(vid_img_path, str(min(vid_len, frame_idx))) for frame_idx in
                                range(curr_idx, curr_idx + (self.sequence_len * self.sample_stride), self.sample_stride)]
        else:
            imgs_paths = None

        return annotations_paths, imgs_paths, ds_name, vid_name, cat_name, int(cat_id), vid_len


@ReaderRegistrar.register
class DataSampleReader(SampleReader):
    def __init__(self, video_dir="datasets/processed/Grouped_frames",
                 annotations_dir="datasets/processed/Annotations",
                 extract_thumbnails=True,
                 thumbnail_image_file="captured_1.jpg",
                 pickle_file="temp/processed.pkl", pickle_mappings=None, mode=None, **kwargs):
        self.short_name = "processed"
        self.video_dir = video_dir
        self.annotations_dir = annotations_dir
        self.extract_thumbnails = extract_thumbnails
        self.thumbnail_image_file = thumbnail_image_file
        self.pickle_mappings = pickle_mappings

        if pickle_mappings is not None:
            self.pickle_mappings = {}
            try:
                for pickle_key, pickle_val in pickle_mappings.items():
                    self.pickle_mappings[pickle_key] = SampleReader(pickle_file=pickle_val, mode="r")
            except:
                print("WARNING: Provided pickle mapping in processed dataset is not found or cannot be processed",
                      pickle_key, pickle_val)

        super().__init__(pickle_file=pickle_file, mode=mode, **kwargs)

    def read_raw(self):
        video_groups = [video_group for video_group in sorted(os.listdir(self.video_dir))]
        video_names = [os.path.join(video_group, video_name) for video_group in video_groups
                       for video_name in sorted(os.listdir(os.path.join(self.video_dir, video_group)))
                       if os.path.isdir(os.path.join(self.video_dir, video_group, video_name))]

        for video_name in tqdm(video_names, desc="Samples Read"):
            try:
                len_frames = len([name for name in os.listdir(os.path.join(self.video_dir, video_name))
                                  if os.path.isdir(os.path.join(self.video_dir, video_name))])
                width, height, thumbnail = extract_width_height_thumbnail_from_image(
                    # 5 is arbitrary to avoid missing initial frames
                    os.path.join(self.video_dir, video_name, "5", self.thumbnail_image_file))

                video_info = {"video_name": os.path.join(self.video_dir, video_name),
                              "video_fps": 25,
                              "video_width": width,
                              "video_height": height,
                              "video_thumbnail": thumbnail}
                audio_info = {"audio_name": '',
                              "has_audio": False}

                # if pickle mappings are provided for preprocessed datasets, then they will update video and audio data
                if self.pickle_mappings is not None:
                    pickled_data = self.pickle_mappings.get(video_name.split("/")[0], None)
                    if pickled_data is not None:
                        try:
                            sample = pickled_data.samples[pickled_data.video_id_to_sample_idx[video_name.split("/")[1]]]
                            update_data = True
                        except:
                            update_data = False
                        if update_data:
                            video_info.update(**{"video_src_name": sample["video_name"],
                                                 "video_src_width": sample["video_width"],
                                                 "video_src_height": sample["video_height"],
                                                 "video_fps": sample["video_fps"],
                                                 "video_thumbnail": sample["video_thumbnail"]})
                            audio_info = {"audio_src_name": sample["audio_name"],
                                          "audio_name": sample["audio_name"],
                                          "has_audio": sample["has_audio"]}

                self.samples.append({"id": video_name,
                                     **video_info, **audio_info,
                                     "len_frames": len_frames,
                                     "annotation_name": os.path.join(self.annotations_dir, video_name),
                                     "annotations": {}
                                     })
                self.video_id_to_sample_idx[video_name] = len(self.samples) - 1
                self.len_frames += self.samples[-1]["len_frames"]
            except:
                print("Error: Access non-existent annotation " + video_name)

    @staticmethod
    def dataset_info():
        return {"summary": "The GASP processed dataset",
                "name": "Processed Dataset",
                "link": "https://github.com/knowledgetechnologyuhh/gasp/tree/GASP_USP_BinDAVE/datasets/processed"}


@SampleRegistrar.register
class DataSample(SampleProcessor):
    def __init__(self, reader, index=-1, frame_index=0, width=640, height=480, **kwargs):
        assert isinstance(reader, DataSampleReader)
        self.short_name = reader.short_name
        self.reader = reader
        self.index = index

        super().__init__(width=width, height=height,
                         video_reader=("ImageCapture", {"extension": "jpg",
                                                        "sub_directories": True,
                                                        "image_file": "captured_1"}), **kwargs)
        if frame_index > 0:
            self.goto_frame(frame_index)
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

    def frames_per_sec(self):
        if self.video_cap is not None:
            return self.reader.samples[self.index]["video_fps"]
        else:
            return 0

    def annotate_frame(self, input_data, plotter,
                       show_gaze=False, show_gaze_label=False, img_names_list=None,
                       **kwargs):
        grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, _ = input_data

        properties = {}

        info = {**info, "frame_annotations": {}}

        grouped_video_frames = {**grouped_video_frames,
                                "PLOT": [["captured"]]
                                }

        try:
            frame_index = self.frame_index()
            frame_name = self.video_cap.frames[frame_index - 1]
            frame_dir = os.path.join(self.video_cap.directory, os.path.dirname(frame_name))
            if grabbed_video and img_names_list is not None:
                for img_name in img_names_list:
                    try:
                        img = cv2.imread(os.path.join(frame_dir, img_name + "_1.jpg"))
                    except cv2.Error:
                        img = np.zeros_like(grouped_video_frames["captured"])

                    grouped_video_frames[img_name] = img
        except:
            pass

        return grabbed_video, grouped_video_frames, grabbed_audio, audio_frames, info, properties

    def get_participant_frame_range(self, participant_id):
        raise NotImplementedError


class DataSplitter(object):
    """
    Reads and writes the split (Train, validation, and Test) sets and stores the groups for training and
    evaluation. The file names are stored in csv files and are not split automatically. This provides an interface
    for manually adding videos to the assigned lists
    """

    def __init__(self, train_csv_file="datasets/processed/train.csv",
                 val_csv_file="datasets/processed/validation.csv",
                 test_csv_file="datasets/processed/test.csv",
                 mode="d", **kwargs):
        if (train_csv_file is None and val_csv_file is None and test_csv_file is None) or mode is None:
            raise AttributeError("Specify atleast 1 csv file and/or choose a supported mode (r,w,x,d)")

        self.train_csv_file = train_csv_file
        self.val_csv_file = val_csv_file
        self.test_csv_file = test_csv_file

        self.mode = mode
        self.columns = ["video_id", "fps", "scene_type", "dataset"]
        self.column_types = {"video_id": "string"}
        self.samples = pd.DataFrame(columns=self.columns + ["split"])
        self.open()

    def read(self, csv_file, split):
        if csv_file is not None:
            if self.mode == "r":  # read or append
                samples = pd.read_csv(csv_file, names=self.columns, header=0, dtype=self.column_types)
                samples["split"] = split

                self.samples = pd.concat([self.samples, samples])
            elif self.mode == "d":  # dynamic: if the pickle_file exists it will be read, otherwise, a new dataset is created
                if os.path.exists(csv_file):
                    samples = pd.read_csv(csv_file, names=self.columns, header=0, dtype=self.column_types)
                    samples["split"] = split
                    self.samples = pd.concat([self.samples, samples])

            elif self.mode == "x":  # safe write
                if os.path.exists(csv_file):
                    raise FileExistsError("Read mode 'x' safely writes a file. "
                                          "Either delete the csv_file '" + csv_file + "' or change the read mode")

    def sample(self, video_id, dataset, fps=0, scene_type=None, split=None, mode="d"):
        # the mode specified here controls the sample whereas the class' mode controls the data splits on file
        # the grouping is based on the video_id and dataset
        if mode == "r":
            match = self.samples[(self.samples["video_id"] == video_id) & (self.samples["dataset"] == dataset)]
            if match.empty:
                match = {"split": None, "scene_type": None}
        elif mode == "d":
            match = self.samples[(self.samples["video_id"] == video_id) & (self.samples["dataset"] == dataset)]
            if match.empty:
                match = pd.DataFrame([[video_id, fps, scene_type, dataset, split]], columns=self.columns + ["split"])
                self.samples = self.samples.append(match, ignore_index=True)
            else:
                if fps is not None:
                    self.samples.loc[
                        (self.samples["video_id"] == video_id) & (self.samples["dataset"] == dataset), "fps"] = fps
                if scene_type is not None:
                    self.samples.loc[(self.samples["video_id"] == video_id) & (
                                self.samples["dataset"] == dataset), "scene_type"] = scene_type
                if split is not None:
                    self.samples.loc[
                        (self.samples["video_id"] == video_id) & (self.samples["dataset"] == dataset), "split"] = split
                match = self.samples[(self.samples["video_id"] == video_id) & (self.samples["dataset"] == dataset)]
        elif mode == "x":
            match = self.samples[(self.samples["video_id"] == video_id) & (self.samples["dataset"] == dataset)]
            if match.empty:
                match = pd.DataFrame([[video_id, fps, scene_type, dataset, split]], columns=self.columns + ["split"])
                self.samples = self.samples.append(match, ignore_index=True)
        elif mode == "w":
            match = pd.DataFrame([[video_id, fps, scene_type, dataset, split]], columns=self.columns + ["split"])
            self.samples = self.samples.append(match, ignore_index=True)

        return match["split"], match["scene_type"]

    def write(self, csv_file, split):
        if csv_file is not None:
            if self.mode == "d" or self.mode == "w" or self.mode == "x":  # read or append
                for name, group in self.samples.groupby("split"):
                    if name == split:
                        group = group.drop(["split"], axis=1)
                        group.to_csv(csv_file, index=False)

    def open(self):
        self.read(self.train_csv_file, "train")
        self.read(self.val_csv_file, "val")
        self.read(self.test_csv_file, "test")

    def save(self):
        self.write(self.train_csv_file, "train")
        self.write(self.val_csv_file, "val")
        self.write(self.test_csv_file, "test")

    def close(self):
        self.save()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self.samples)


class DataWriter(object):
    """
    Writes the dataset in the format supported by the DataLoader. This writer assumes the structures of
    grouped_video_frames_list and info_list as resulting from annotate_frames in VideoProcessor
    """

    def __init__(self, dataset_name, video_name, save_dir="datasets/processed/",
                 output_video_size=(640, 480), frames_per_sec=20,
                 write_images=True, write_videos=False, write_annotations=True):

        if dataset_name == "processed":
            rename_ds = video_name.split(os.sep)
            dataset_name = rename_ds[0]
            video_name = rename_ds[1]

        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.video_name = video_name
        self.output_video_size = output_video_size
        self.frames_per_sec = frames_per_sec
        self.write_images = write_images
        self.write_videos = write_videos
        self.write_annotations = write_annotations
        # loading annotations not needed if they will not be written
        if write_annotations:
            path_to_pickle = os.path.join(self.save_dir, "Annotations", self.dataset_name, self.video_name)
            if os.path.exists(path_to_pickle):
                self.annotations = {}
                for root_dir, dirnames, filenames in sorted(os.walk(path_to_pickle)):
                    for filename in filenames:
                        if filename.endswith(".pkl"):
                            self.annotations[int(filename.rstrip(".pkl"))] = pickle.load(
                                open(os.path.join(path_to_pickle,
                                                  filename), "rb"))
            else:
                self.annotations = {}
        else:
            self.annotations = {}
        self.videos = {}

    def add_detections(self, returns, models):
        current_dict = {}
        for idx_model, model_data in enumerate(models):
            for i, frame_dict in enumerate(returns[2 + idx_model][4]):
                if self.write_annotations:
                    current_dict = {**current_dict, **self.make_id_key(frame_dict)}
                if self.write_images or self.write_videos:
                    for image_group in returns[2 + idx_model][1][i]["PLOT"]:
                        for image_name in image_group:
                            image = returns[2 + idx_model][1][i][image_name]
                            if image is not None and np.shape(image) != () and image.any():
                                self.write_transformed_image(frame_id=frame_dict["frame_info"]["frame_id"],
                                                             img_array=image, img_name=image_name)
        if self.write_annotations:
            self.merge_into_annotations(current_dict)

    def merge_into_annotations(self, current_dict):
        for key in current_dict:
            if key in self.annotations and not "frame_info":
                self.annotations[key] = self.deep_append(self.annotations[key], current_dict[key])
            else:
                self.annotations[key] = current_dict[key]

    @staticmethod
    def make_id_key(old_dict):
        # quick restructuring so that frame_id is top level
        new_dict = {old_dict["frame_info"]["frame_id"]: old_dict}
        return new_dict

    @staticmethod
    def deep_append(old_dict, new_dict):
        # merges dicts of dicts (of dicts) of lists without deleting
        for key in old_dict:
            if isinstance(old_dict[key], dict):
                if new_dict[key] and new_dict[key] is not None:
                    old_dict[key] = DataWriter.deep_append(old_dict[key], new_dict[key])
            else:
                if new_dict[key] and new_dict[key] is not None:
                    old_dict[key].extend(new_dict[key])
        return old_dict

    def write_transformed_image(self, frame_id, img_array, img_name):
        # save the transformed images as jpegs, see mattermost for folder structure
        write_path = os.path.join(self.save_dir, "{}", self.dataset_name)
        if self.write_images:
            img_path = os.path.join(write_path.format("Grouped_frames"), self.video_name, str(frame_id))
            if not os.path.exists(img_path):
                os.makedirs(img_path, exist_ok=True)
            index = 1
            while os.path.exists(os.path.join(img_path, img_name + "_" + str(index) + ".jpg")):
                index += 1
            cv2.imwrite(os.path.join(img_path, img_name + "_" + str(index) + ".jpg"), img_array)
        if self.write_videos:
            if img_name in self.videos:
                pass
            else:
                vid_path = os.path.join(write_path.format("Videos"), self.video_name)
                if not os.path.exists(vid_path):
                    os.makedirs(vid_path, exist_ok=True)
                video_enc = cv2.VideoWriter_fourcc(*"XVID")
                self.videos[img_name] = {"writer": cv2.VideoWriter(os.path.join(vid_path, img_name + '.avi'), video_enc,
                                                                   self.frames_per_sec, self.output_video_size),
                                         # 25, (1232,504)), #
                                         "last_frame": frame_id - 1}
            if self.videos[img_name]["last_frame"] < frame_id:
                # self.videos[img_name]["writer"].write(cv2.resize(img_array, (1232,504)))
                self.videos[img_name]["writer"].write(cv2.resize(img_array, self.output_video_size))
                self.videos[img_name]["last_frame"] = frame_id

    def dump_annotations(self):
        if self.write_annotations:
            path_to_pickle = os.path.join(self.save_dir, "Annotations", self.dataset_name, self.video_name)
            if not os.path.exists(path_to_pickle):
                os.makedirs(path_to_pickle, exist_ok=True)
            for frame, annotation in self.annotations.items():
                pickle.dump(annotation, open(os.path.join(path_to_pickle, str(frame) + ".pkl"), "wb"),
                            protocol=pickle.HIGHEST_PROTOCOL)

    def clear_annotations(self):
        self.annotations = {}

    def dump_videos(self):
        if self.write_videos:
            for vid_name in self.videos.keys():
                self.videos[vid_name]["writer"].release()
            self.videos = {}

    def set_new_name(self, vid_name, output_vid_size=None, fps=None):
        if os.sep in vid_name:
            rename_ds = vid_name.split(os.sep)
            self.dataset_name = rename_ds[0]
            self.video_name = rename_ds[1]
        else:
            self.video_name = vid_name
        self.clear_annotations()
        if output_vid_size is not None:
            self.output_video_size = output_vid_size
        if fps is not None:
            self.frames_per_sec = fps
