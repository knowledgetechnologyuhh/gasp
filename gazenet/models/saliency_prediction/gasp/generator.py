import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

try:  # leverages intel IPP for accelerated image loading
    from accimage import Image
    torchvision.set_image_backend('accimage')
except:
    from PIL import Image

INP_IMG_MODE = "RGB"
TRG_IMG_MODE = "L"
DEFAULT_IMG_SIDE = 10 # just a place-holder


def load_video_frames(grouped_frames_list, last_frame_idx, inp_img_names_list, img_mean, img_std, img_width, img_height, frames_len=16):
    # load video video_frames_list, process them and return a suitable tensor
    frame_number = last_frame_idx
    start_frame_number = frame_number - frames_len # + 1
    start_frame_number = max(0, start_frame_number)
    frames_list_idx = [f for f in range(start_frame_number, frame_number)]
    if len(frames_list_idx) < frames_len:
        nsh = frames_len - len(frames_list_idx)
        frames_list_idx = np.concatenate((np.tile(frames_list_idx[0], (nsh)), frames_list_idx), axis=0)
    frames = [[] for _ in inp_img_names_list]

    for i in range(len(frames_list_idx)):
        for frame_name, frame in grouped_frames_list[i].items():
            if not frame_name in inp_img_names_list:
                continue
            else:
                group_idx = inp_img_names_list.index(frame_name)
                # TODO (fabawi): loading the first frame on failure is not ideal. Find a better way
            try:
                img = cv2.resize(frame.copy(), (img_width, img_height))
            except:
                try:
                    img = cv2.resize(frame.copy(), (img_width, img_height))
                except:
                    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = F.to_tensor(img)
            img = F.normalize(img, img_mean, img_std)
            frames[group_idx].append(img)
    frames = [torch.stack(frames[group_idx], dim=0) for group_idx in range(len(inp_img_names_list))]
    frames = torch.cat(frames, 1) # .permute(1, 0, 2, 3)
    if len(frames_list_idx) > 1:
        return frames
    else:
        return frames[0, ...]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class GASPDataset(Dataset):
    """
    Reads the DataWriter-generated dataset as pytorch compatible objects. When sequence_len is not 1,
    any non consecutive images will be zero padded. Loads images in the form
    <batch_size, sequence_len(optional), channels, height, width>
    """
    def __init__(self, csv_file, video_dir,
                 inp_img_names_list, gt_img_names_list,
                 inp_img_transform=None, gt_img_transform=None,
                 sequence_len=1, exhaustive=False, cache_sequence_pointers=False
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with the sample descriptions
            video_dir (string): Directory with all the frames separated into folders
            inp_img_names_list (list): List of the input image prefixes
            gt_img_names_list (list): List of the ground-truth images prefixes
            inp_img_transform (callable, optional): Optional transform applied to the input image
            gt_img_transform (callable, optional): Optional transform applied to the ground-truth image
            sequence_len (int): Length of an image sequence. If None, the sequence dimension is removed
            exhaustive (boolean): Return gt images for the entire sequence. Otherwise, only the last gt image
            cache_sequence_pointers (boolean): Remember the last sample frame and resume. Otherwise, samples randomly
        """
        self.samples = pd.read_csv(csv_file)
        self.samples = self.samples[(self.samples["scene_type"] == "Social")].reset_index()
        self.video_dir = video_dir
        self.input_img_names_list = inp_img_names_list
        self.gt_img_names_list = gt_img_names_list
        self.input_img_transform = inp_img_transform
        self.gt_img_transform = gt_img_transform
        self.sequence_len = sequence_len
        self.exhaustive = exhaustive
        self.cache_sequence_pointers = cache_sequence_pointers
        self.sequence_pointer_cache = {dataset: {} for dataset in self.samples.dataset.unique()}  # {dataset_name: video_name: (curr_idx, min_idx, max_idx)}

    def len_frames(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getfilenames__(self, idx):
        ds_name = self.samples.loc[idx, "dataset"]
        vid_name = self.samples.loc[idx, "video_id"]
        vid_path = os.path.join(self.video_dir, ds_name, vid_name)

        curr_idx, min_idx, max_idx = self.sequence_pointer_cache[ds_name].get(vid_name, (1, 1, len(os.listdir(vid_path))))
        if self.cache_sequence_pointers:
            next_idx = curr_idx + self.sequence_len
            self.sequence_pointer_cache[ds_name][vid_name] = (next_idx if next_idx < max_idx else 1, min_idx, max_idx)
        else:
            curr_idx = random.randint(min_idx, max_idx)
        imgs_paths = [os.path.join(vid_path, str(frame_idx)) for frame_idx in range(curr_idx, curr_idx + self.sequence_len)]

        return imgs_paths

    def __getimgs__(self, idx):
        all_input_imgs = []
        all_gt_imgs_dict = {gt_img_name: [] for gt_img_name in self.gt_img_names_list}
        if self.exhaustive:
            all_gt_imgs_dict.update(**{"seq_"+gt_img_name: [] for gt_img_name in self.gt_img_names_list})

        imgs_paths = self.__getfilenames__(idx)

        prev_input_imgs = [None] * len(self.input_img_names_list)
        for seq_img_idx, seq_img_path in enumerate(imgs_paths):

            input_imgs = None
            for input_img_idx, input_img_name in enumerate(self.input_img_names_list):
                try:
                    input_img = Image.open(os.path.join(seq_img_path, input_img_name + "_1.jpg")).convert(INP_IMG_MODE)
                    prev_input_imgs[input_img_idx] = input_img.copy()
                except:
                    if prev_input_imgs[input_img_idx] is None:
                        input_img = Image.new(INP_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                        input_img.putpixel((0, 0), tuple([1] * len(input_img.getpixel((0, 0)))))  # to avoid nan
                    else:
                        input_img = prev_input_imgs[input_img_idx].copy()
                if self.input_img_transform:
                    input_img = self.input_img_transform(input_img)
                input_img = input_img if torch.is_tensor(input_img) else torchvision.transforms.ToTensor()(input_img)
                input_imgs = torch.cat([input_imgs, input_img]) if input_imgs is not None else input_img
            all_input_imgs.append(input_imgs)

            for gt_img_name in self.gt_img_names_list:

                try:
                    gt_img = Image.open(os.path.join(seq_img_path, gt_img_name + "_1.jpg")).convert(TRG_IMG_MODE)
                    if self.exhaustive:
                        gt_img_seq = gt_img.copy()
                    else:
                        gt_img_seq = None
                except:
                    if self.exhaustive:
                        gt_img_seq = Image.new(TRG_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                        gt_img_seq.putpixel((0, 0), 1)
                        if not all_gt_imgs_dict[gt_img_name] and (seq_img_idx == len(imgs_paths) - 1):
                            gt_img = Image.new(TRG_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                            gt_img.putpixel((0, 0), 1)  # to avoid nan
                        else:
                            gt_img = None
                    elif not all_gt_imgs_dict[gt_img_name] and (seq_img_idx == len(imgs_paths) - 1):
                        gt_img_seq = None
                        gt_img = Image.new(TRG_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                        gt_img.putpixel((0, 0), 1)  # to avoid nan
                    else:
                        gt_img_seq = None
                        gt_img = None
                if gt_img is not None:
                    if self.gt_img_transform:
                        gt_img = self.gt_img_transform(gt_img)
                    gt_img = gt_img if torch.is_tensor(gt_img) else torchvision.transforms.ToTensor()(gt_img)
                if gt_img_seq is not None:
                    if self.gt_img_transform:
                        gt_img_seq = self.gt_img_transform(gt_img_seq)
                    gt_img_seq = gt_img_seq if torch.is_tensor(gt_img_seq) else torchvision.transforms.ToTensor()(gt_img_seq)

                if self.exhaustive:
                    all_gt_imgs_dict["seq_" + gt_img_name].append(gt_img_seq)
                    if gt_img is not None:
                        all_gt_imgs_dict[gt_img_name] = [gt_img]
                elif gt_img is not None:
                    all_gt_imgs_dict[gt_img_name] = [gt_img]

        all_input_imgs = torch.cat(all_input_imgs) if self.sequence_len == 1 else torch.stack(all_input_imgs)
        for k_gt_imgs, v_gt_imgs in all_gt_imgs_dict.items():
            if k_gt_imgs.startswith("seq_"):
                all_gt_imgs_dict[k_gt_imgs] = torch.stack(v_gt_imgs)
                # all_gt_imgs_dict[k_gt_imgs] = torch.cat(v_gt_imgs) if self.sequence_len == 1 or not self.exhaustive else torch.stack(v_gt_imgs)
            else:
                all_gt_imgs_dict[k_gt_imgs] = torch.cat(v_gt_imgs)

        return all_input_imgs, all_gt_imgs_dict

    def __getanno__(self, idx):
        raise NotImplementedError("Annotations not needed since the images are already transformed for all models. "
                                  "Might be necessary for other tasks w/o multitask learning or other modality inputs.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.__getimgs__(idx)
