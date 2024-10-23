import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as Fv
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl


from gazenet.utils.registrar import *
from gazenet.utils.audio_features import MFCCAudioFeatures
from gazenet.utils.dataset_processors import BaseDataset

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
        u_inp_img_names_list = set(inp_img_names_list)
        for frame_name, frame in grouped_frames_list[i].items():
            if not frame_name in inp_img_names_list:
                continue
            else:
                u_inp_img_names_list.remove(frame_name)
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
            img = Fv.to_tensor(img)
            img = Fv.normalize(img, img_mean, img_std)
            frames[group_idx].append(img)
        for miss_inp_img in u_inp_img_names_list:
            miss_inp_idx = inp_img_names_list.index(miss_inp_img)
            frames[miss_inp_idx].append(frames[miss_inp_idx][-1].clone())
    frames = [torch.stack(frames[group_idx], dim=0) for group_idx in range(len(inp_img_names_list))]
    frames = torch.cat(frames, 1)  # .permute(1, 0, 2, 3)
    if len(frames_list_idx) > 1:
        return frames
    else:
        return frames[0, ...]


class GASPData(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=16, sequence_len=1,
                 trg_img_width=60, trg_img_height=60,
                 inp_img_width=120, inp_img_height=120,
                 inp_img_mean=(110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0),
                 inp_img_std=(38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0),
                 train_dataset_properties=None, val_dataset_properties=None, test_dataset_properties=None, **kwargs):
        super(GASPData, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_len = sequence_len

        self.trg_img_width = trg_img_width
        self.trg_img_height = trg_img_height
        self.inp_img_width = inp_img_width
        self.inp_img_height = inp_img_height
        self.inp_img_mean = inp_img_mean
        self.inp_img_std = inp_img_std

        self.train_dataset_properties = train_dataset_properties
        self.val_dataset_properties = val_dataset_properties
        self.test_dataset_properties = test_dataset_properties

        self.exhaustive = False

    def prepare_data(self):

        # dataset properties
        if self.train_dataset_properties is None:
            self.train_dataset_properties = {"csv_file": "datasets/processed/train.csv",
                                             "video_dir": "datasets/processed/Grouped_frames",
                                             "inp_img_names_list": ["det_transformed_dave", "det_transformed_esr9",
                                                                      "det_transformed_vidgaze",
                                                                      "det_transformed_gaze360"],
                                             "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"],
                                             "gt_mappings": {"gt_salmap": "transformed_salmap",
                                                             "gt_fixmap": "transformed_fixmap",
                                                             }}
        else:
            self.train_dataset_properties.setdefault("gt_mappings", {"gt_salmap": "transformed_salmap",
                                                                     "gt_fixmap": "transformed_fixmap"})

        if self.val_dataset_properties is None:
            self.val_dataset_properties = {"csv_file": "datasets/processed/validation.csv",
                                           "video_dir": "datasets/processed/Grouped_frames",
                                           "inp_img_names_list": ["det_transformed_dave", "det_transformed_esr9",
                                                                    "det_transformed_vidgaze",
                                                                    "det_transformed_gaze360"],
                                           "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"],
                                           "gt_mappings": {"gt_salmap": "transformed_salmap",
                                                           "gt_fixmap": "transformed_fixmap",
                                                           }}
        else:
            self.val_dataset_properties.setdefault("gt_mappings", {"gt_salmap": "transformed_salmap",
                                                                   "gt_fixmap": "transformed_fixmap"})

        if self.test_dataset_properties is None:
            self.test_dataset_properties = {"csv_file": "datasets/processed/test.csv",
                                            "video_dir": "datasets/processed/Grouped_frames",
                                            "inp_img_names_list": ["det_transformed_dave", "det_transformed_esr9",
                                                                     "det_transformed_vidgaze",
                                                                     "det_transformed_gaze360"],
                                            "gt_img_names_list": ["transformed_salmap", "transformed_fixmap"],
                                            "gt_mappings": {"gt_salmap": "transformed_salmap",
                                                            "gt_fixmap": "transformed_fixmap",
                                                            }}
        else:
            self.test_dataset_properties.setdefault("gt_mappings", {"gt_salmap": "transformed_salmap",
                                                                    "gt_fixmap": "transformed_fixmap"})

        # transforms for images
        input_img_transform = transforms.Compose([
            transforms.Resize((self.inp_img_height, self.inp_img_width)),
            transforms.ToTensor(),
            transforms.Normalize(self.inp_img_mean, self.inp_img_std),
        ])
        gt_img_transform = transforms.Compose([
            transforms.Resize((self.trg_img_height, self.trg_img_width)),
            transforms.ToTensor()
        ])

        self.train_dataset_properties.update(sequence_len=self.sequence_len, gt_img_transform=gt_img_transform,
                                            inp_img_transform=input_img_transform, exhaustive=self.exhaustive)
        self.train_dataset = GASPDataset(**self.train_dataset_properties)

        self.val_dataset_properties.update(sequence_len=self.sequence_len, gt_img_transform=gt_img_transform,
                                          inp_img_transform=input_img_transform, exhaustive=self.exhaustive)
        self.val_dataset = GASPDataset(**self.val_dataset_properties)

        self.test_dataset_properties.update(sequence_len=self.sequence_len, gt_img_transform=gt_img_transform,
                                           inp_img_transform=input_img_transform, exhaustive=self.exhaustive)
        self.test_dataset = GASPDataset(**self.test_dataset_properties)

        # self.train_dataset, self.val_dataset = random_split(self.train_dataset, [55000, 5000])
        # update the batch size based on the train dataset if experimenting on small dummy data
        if len(self.train_dataset) < self.batch_size:
            self.batch_size = len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @staticmethod
    def update_infer_config(log_path, checkpoint_file, train_config, infer_config, device):
        # if isinstance(device, list):
        #     infer_config.device = "cuda:" + str(device[0])
        if isinstance(device, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device)
        elif isinstance(device, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = device

        # TODO (fabawi): the last checkpoint isn't always the best but this is the safest option for now
        # update the datasplitter to include test file specified in the trainer
        infer_config.datasplitter_properties.update(test_csv_file=train_config.test_dataset_properties["csv_file"])
        # update the metrics file path
        infer_config.metrics_save_file = os.path.join(log_path, "metrics.csv")
        # update the sampling names list. This is specific to DataSample
        infer_config.sampling_properties.update(
            img_names_list=train_config.test_dataset_properties["inp_img_names_list"] +
                           train_config.test_dataset_properties["gt_img_names_list"])

        # update the targeted model properties
        for mod_group in infer_config.model_groups:
            for mod in mod_group:
                if not train_config.inferer_name == mod[0]:
                    continue

                # try extracting the window size if it exists, otherwise, assume single frame
                for w_size_name in ["w_size", "frames_len", "sequence_len"]:
                    if w_size_name in train_config.model_properties:
                        if mod[1] == -1:
                            mod[1] = train_config.model_properties[w_size_name]
                        if mod[2] == -1:
                            mod[2] = [mod[1] - 1]
                        break
                if mod[1] == -1:
                    mod[1] = 1
                if mod[2] == -1:
                    mod[2] = [0]

                # infer the frames_len from sequence_len
                if "frames_len" not in train_config.model_properties and "sequence_len" in train_config.model_properties:
                    train_config.model_properties["frames_len"] = train_config.model_properties["sequence_len"]

                # update the configuration
                mod[3].update(**train_config.model_properties,
                              weights_file=checkpoint_file,
                              model_name=train_config.model_name)

                # update the input image names list. This is specific to gasp
                if mod[4]["inp_img_names_list"] is None:
                    mod[4].update(inp_img_names_list=train_config.test_dataset_properties["inp_img_names_list"])

                break

        return infer_config


@ModelDataRegistrar.register
class GASPEncGMUALSTMConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPEncGMUALSTMConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPEncALSTMGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPEncALSTMGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPEncALSTMConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPEncALSTMConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPDAMEncGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPDAMEncGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPDAMEncALSTMGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPDAMEncALSTMGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPDAMEncGMUALSTMConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPDAMEncGMUALSTMConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPEncConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPEncConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPEncAddConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPEncAddConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPSEEncConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPSEEncConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class GASPEncGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(GASPEncGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class SequenceGASPEncALSTMConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPEncALSTMConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class SequenceGASPEncALSTMGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPEncALSTMGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class SequenceGASPEncGMUALSTMConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPEncGMUALSTMConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class SequenceGASPDAMEncGMUALSTMConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPDAMEncGMUALSTMConvData, self).__init__(*args, **kwargs)
        self.exhaustive = True


@ModelDataRegistrar.register
class SequenceGASPDAMEncALSTMGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPDAMEncALSTMGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = True


@ModelDataRegistrar.register
class SequenceGASPEncRGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPEncRGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = False


@ModelDataRegistrar.register
class SequenceGASPDAMEncRGMUConvData(GASPData):
    def __init__(self, *args, **kwargs):
        super(SequenceGASPDAMEncRGMUConvData, self).__init__(*args, **kwargs)
        self.exhaustive = True


class GASPDataset(BaseDataset):
    """
    Reads the DataWriter-generated dataset as pytorch compatible objects. When sequence_len is not 1,
    any non consecutive images will be zero padded. Loads images in the form
    <batch_size, sequence_len(optional), channels, height, width>
    """
    def __init__(self, gt_img_names_list, gt_img_transform=None, exhaustive=False, **kwargs):
        """
        Args:
            gt_img_names_list (list): List of the ground-truth images prefixes
            gt_img_transform (callable, optional): Optional transform applied to the ground-truth image
            exhaustive (boolean, optional): Return gt images for the entire sequence. Otherwise, only the last gt image
        """
        kwargs.update({"annotation_dir": "", "replicate_class_samples": False, "classes": []})
        super(GASPDataset, self).__init__(**kwargs)

        self.gt_img_names_list = gt_img_names_list
        self.gt_img_transform = gt_img_transform
        self.exhaustive = exhaustive

        self.audio_encoder = MFCCAudioFeatures()

    @staticmethod
    def get_attributes():
        return {}

    def __getimgs__(self, idx):
        all_input_imgs = []
        all_gt_imgs_dict = {gt_img_name: [] for gt_img_name in self.gt_img_names_list}
        if self.exhaustive:
            all_gt_imgs_dict.update(**{"seq_"+gt_img_name: [] for gt_img_name in self.gt_img_names_list})

        _, imgs_paths, ds_name, vid_name, _, _, _ = self.__getfilenames__(idx)

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
                        input_img.putpixel((0, 0), tuple([1] * len(input_img.getpixel((0, 0)))))
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
                        if not all_gt_imgs_dict[gt_img_name] and (seq_img_idx == len(imgs_paths) - 1):
                            gt_img = Image.new(TRG_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                            gt_img.putpixel((0, 0), 1)
                        else:
                            gt_img = None
                    elif not all_gt_imgs_dict[gt_img_name] and (seq_img_idx == len(imgs_paths) - 1):
                        gt_img_seq = None
                        gt_img = Image.new(TRG_IMG_MODE, (DEFAULT_IMG_SIDE, DEFAULT_IMG_SIDE))
                        gt_img.putpixel((0, 0), 1)
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.__getimgs__(idx)
