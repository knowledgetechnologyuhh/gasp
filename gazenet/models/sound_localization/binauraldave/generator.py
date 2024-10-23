import os
import random
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as Fv
from torch.utils.data import Dataset
import torchvision
import librosa as sf

try:  # leverages intel IPP for accelerated image loading
    from accimage import Image
    torchvision.set_image_backend('accimage')
except:
    from PIL import Image

INP_IMG_MODE = "RGB"
TRG_IMG_MODE = "L"
DEFAULT_IMG_SIDE = 10  # just a place-holder


def create_audio_packet(in_data, frame_number, frames_len=16):
    n_frame = in_data.shape[0]
    # if the frame number is larger, we just use the last sound one heard
    frame_number = min(frame_number, n_frame)
    starting_frame = frame_number - frames_len + 1
    # ensure we do not have any negative video_frames_list
    starting_frame = max(0, starting_frame)
    data_pack = in_data[starting_frame:frame_number+1, :, :]
    n_pack = data_pack.shape[0]

    if n_pack < frames_len:
        nsh = frames_len - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0,:,:], (nsh, 1, 1)), data_pack), axis=0)

    assert data_pack.shape[0] == frames_len
    data_pack = np.tile(data_pack, (3, 1, 1, 1))
    return data_pack


def create_data_packet(in_data, frame_number, frames_len=16):
    in_data = np.array(in_data)
    n_frame_left = in_data[0]
    n_frame_right = in_data[1]
    data_pack_left = create_audio_packet(n_frame_left, frame_number, frames_len=frames_len)
    data_pack_right = create_audio_packet(n_frame_right, frame_number, frames_len=frames_len)
    return (data_pack_left, data_pack_right), frame_number


def get_wav_features(features, frame_number, frames_len=16):
    audio_data, valid_frame_number = create_data_packet(features, frame_number, frames_len=frames_len)
    return torch.from_numpy(audio_data[0]).float(), torch.from_numpy(audio_data[1]).float(), valid_frame_number


def load_video_frames(frames_list, last_frame_idx, valid_frame_idx, img_mean, img_std, img_width, img_height, frames_len=16):
    # load video video_frames_list, process them and return a suitable tensor
    frame_number = min(last_frame_idx, valid_frame_idx)
    start_frame_number = frame_number - frames_len + 1
    start_frame_number = max(0, start_frame_number)
    frames_list_idx = [f for f in range(start_frame_number, frame_number)]
    if len(frames_list_idx) < frames_len:
        nsh = frames_len - len(frames_list_idx)
        frames_list_idx = np.concatenate((np.tile(frames_list_idx[0], (nsh)), frames_list_idx), axis=0)
    frames = []
    for i in range(len(frames_list_idx)):
        try:
            img = cv2.resize(frames_list[frames_list_idx[i]].copy(), (img_width, img_height))
        except:
            try:
                img = cv2.resize(frames_list[frames_list_idx[0]].copy(), (img_width, img_height))
            except:
                img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.convert('RGB')
        img = Fv.to_tensor(img)
        img = Fv.normalize(img, img_mean, img_std)
        frames.append(img)
    frames = torch.stack(frames, dim=0)
    return frames.permute([1, 0, 2, 3])


class BinauralDAVEDataset(Dataset):
    """
    Reads the DataWriter-generated dataset as pytorch compatible objects. When frames_len is not 1,
    any non consecutive images will be zero padded. Loads images in the form
    <batch_size, frames_len(optional), channels, height, width>
    """
    def __init__(self, csv_file, video_dir, annotation_dir,
                 inp_img_name, gt_img_name,
                 inp_img_transform=None, gt_img_transform=None,
                 frames_len=16, sample_stride=2, random_flips=True,
                 audiovisual=True
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with the sample descriptions
            video_dir (string): Directory with all the frames separated into folders
            annotation_dir (string): Directory with all the frame annotations separated into files (should contain MFCC audio features)
            inp_img_name (string): Input image prefix
            gt_img_name (string): Ground-truth images prefix
            inp_img_transform (callable, optional): Optional transform applied to the input image
            gt_img_transform (callable, optional): Optional transform applied to the ground-truth image
            frames_len (int): Length of an image sequence. If None, the sequence dimension is removed
            sample_stride (boolean): Skip n-between frames per sample
            random_flips (boolean): Randomly flip the images and sounds horizontally
            audiovisual (boolean): Loads audio data
        """
        self.samples = pd.read_csv(csv_file, dtype={"video_id": "string"}).reset_index()
        # self.samples = self.samples[(self.samples["scene_type"] == "Social")].reset_index()
        self.video_dir = video_dir
        self.annotation_dir = annotation_dir
        self.input_img_name = inp_img_name
        self.gt_img_name = gt_img_name
        self.input_img_transform = inp_img_transform
        self.gt_img_transform = gt_img_transform
        self.frames_len = frames_len
        self.sample_stride = sample_stride
        self.random_flips = random_flips
        self.audiovisual = audiovisual

        self.enumerate_video_frames()

    def enumerate_video_frames(self):

        self.samples["frames_len"] = self.samples.apply(lambda row:
                                                        len(os.listdir(os.path.join(self.annotation_dir, row["dataset"],
                                                                                    row["video_id"]))), axis=1)
        self.samples["frame_pointer"] = self.samples["frames_len"].rolling(min_periods=1,
                                                                           window=len(self.samples)).sum()

    def __len__(self):
        return int(self.samples["frame_pointer"].iloc[-1])

    def __getfilenames__(self, global_idx):
        idx = self.samples.index[self.samples["frame_pointer"] > global_idx].tolist()[0]

        ds_name = self.samples.loc[idx, "dataset"]
        vid_name = self.samples.loc[idx, "video_id"]
        vid_len = self.samples.loc[idx, "frames_len"]

        curr_idx = int(global_idx - self.samples.loc[idx, "frame_pointer"] + self.samples.loc[idx, "frames_len"] + 1)

        vid_img_path = os.path.join(self.video_dir, ds_name, vid_name)
        imgs_paths = [os.path.join(vid_img_path, str(min(vid_len, frame_idx))) for frame_idx in
                      range(curr_idx, curr_idx + (self.frames_len * self.sample_stride), self.sample_stride)]

        if self.audiovisual:
            vid_anno_path = os.path.join(self.annotation_dir, ds_name, vid_name)
            annotations_paths = [os.path.join(vid_anno_path, str(min(vid_len, frame_idx))) for frame_idx in
                                 range(curr_idx, curr_idx + (self.frames_len * self.sample_stride), self.sample_stride)]
            return annotations_paths, imgs_paths, ds_name, vid_name, vid_len
        else:
            return imgs_paths, ds_name, vid_name, vid_len

    def __getimgs__(self, idx):
        if self.random_flips:
            flip_h = random.random()
            flip_h = True if flip_h < 0.5 else False
        else:
            flip_h = False

        all_input_imgs = []
        all_gt_imgs_dict = {self.gt_img_name: []}

        if self.audiovisual:
            all_input_auds = None
            all_input_auds_left = torch.from_numpy(
                np.tile(np.zeros((self.frames_len, 64, 64), dtype=np.float32), (3, 1, 1, 1)))
            all_input_auds_right = torch.zeros_like(all_input_auds_left)

            annotations_paths, imgs_paths, ds_name, vid_name, vid_len = self.__getfilenames__(idx)
        else:
            imgs_paths, ds_name, vid_name, vid_len = self.__getfilenames__(idx)
            annotations_paths = [None] * len(imgs_paths)

        prev_input_imgs = None
        for seq_idx, (seq_ann_path, seq_img_path) in enumerate(zip(annotations_paths, imgs_paths)):
            if self.audiovisual:
                try:
                    if all_input_auds is None:
                        with open(seq_ann_path + ".pkl", "rb") as fp:
                            temp_annotations = pickle.load(fp)
                            all_input_auds = temp_annotations["frame_detections_greedygaze"]["audio_features"]
                            all_input_auds = all_input_auds[0]

                            if all_input_auds is None:
                                # check the surrounding annotations
                                ann_path_dir = seq_ann_path[:seq_ann_path.rindex("/")]
                                for ann_idx in range(int(seq_ann_path.split("/")[-1]) - self.sample_stride//2,
                                                      int(seq_ann_path.split("/")[-1]) + self.sample_stride//2):
                                    try:
                                        with open(os.path.join(ann_path_dir, str(ann_idx) + ".pkl"), "rb") as fp:
                                            temp_annotations = pickle.load(fp)
                                            all_input_auds = temp_annotations["frame_detections_greedygaze"]["audio_features"]
                                            all_input_auds = all_input_auds[0]
                                            if all_input_auds is not None:
                                                all_input_auds_left = torch.from_numpy(np.tile(all_input_auds[0, :self.frames_len, ...], (3, 1, 1, 1))).type(torch.float32)
                                                all_input_auds_right = torch.from_numpy(np.tile(all_input_auds[1, :self.frames_len, ...], (3, 1, 1, 1))).type(torch.float32)
                                                continue
                                    except:
                                        pass
                            else:
                                all_input_auds_left = torch.from_numpy(np.tile(all_input_auds[0, :self.frames_len, ...], (3, 1, 1, 1))).type(torch.float32)
                                all_input_auds_right = torch.from_numpy(np.tile(all_input_auds[1, :self.frames_len, ...], (3, 1, 1, 1))).type(torch.float32)
                    else:
                        pass
                except:
                    pass

            input_imgs = None
            try:
                input_img = Image.open(os.path.join(seq_img_path, self.input_img_name + "_1.jpg")).convert(INP_IMG_MODE)
                prev_input_imgs = input_img.copy()
            except:
                if prev_input_imgs is None:
                    input_img = Image.new(INP_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                    input_img.putpixel((0, 0), tuple([1] * len(input_img.getpixel((0, 0)))))
                else:
                    input_img = prev_input_imgs.copy()
            if self.input_img_transform:
                input_img = self.input_img_transform(input_img)
            input_img = input_img if torch.is_tensor(input_img) else torchvision.transforms.ToTensor()(input_img)
            if flip_h:
                input_img = torchvision.transforms.RandomHorizontalFlip(p=2)(input_img)
            input_imgs = torch.cat([input_imgs, input_img]) if input_imgs is not None else input_img
            all_input_imgs.append(input_imgs)

            try:
                gt_img = Image.open(os.path.join(seq_img_path, self.gt_img_name + "_1.jpg")).convert(TRG_IMG_MODE)
            except:
                if not all_gt_imgs_dict[self.gt_img_name] and (seq_idx == len(imgs_paths) - 1):
                    gt_img = Image.new(TRG_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                    gt_img.putpixel((0, 0), 1)
                else:
                    gt_img = None
            if gt_img is not None:
                if self.gt_img_transform:
                    gt_img = self.gt_img_transform(gt_img)
                gt_img = gt_img if torch.is_tensor(gt_img) else torchvision.transforms.ToTensor()(gt_img)
                if flip_h:
                    gt_img = torchvision.transforms.RandomHorizontalFlip(p=2)(gt_img)

                all_gt_imgs_dict[self.gt_img_name] = [gt_img]

        all_input_imgs = torch.cat(all_input_imgs) if self.frames_len == 1 else torch.stack(all_input_imgs)
        for k_gt_imgs, v_gt_imgs in all_gt_imgs_dict.items():
            all_gt_imgs_dict[k_gt_imgs] = torch.cat(v_gt_imgs)

        if self.audiovisual:
            if flip_h:
                temp_auds = all_input_auds_left
                all_input_auds_left = all_input_auds_right
                all_input_auds_right = temp_auds

            return all_input_imgs.permute([1, 0, 2, 3]), all_input_auds_left, all_input_auds_right, all_gt_imgs_dict
        else:
            return all_input_imgs.permute([1, 0, 2, 3]), all_gt_imgs_dict

    def __getanno__(self, idx):
        raise NotImplementedError("Annotations not needed since the images are already transformed for all models. "
                                  "Might be necessary for other tasks w/o multitask learning or other modality inputs.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.__getimgs__(idx)
