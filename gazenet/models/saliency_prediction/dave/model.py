#
# DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction
# https://arxiv.org/abs/1905.10693
# https://hrtavakoli.github.io/DAVE/
#
# Copyright by Hamed Rezazadegan Tavakoli
#

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from gazenet.models.shared_components.resnet3d.model import resnet18


class ScaleUp(nn.Module):

    def __init__(self, in_size, out_size):
        super(ScaleUp, self).__init__()

        self.combine = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)

        self._weights_init()

    def _weights_init(self):

        nn.init.kaiming_normal_(self.combine.weight)
        nn.init.constant_(self.combine.bias, 0.0)

    def forward(self, inputs):
        output = F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.combine(output)
        output = F.relu(output, inplace=True)
        return output


class DAVE(nn.Module):

    def __init__(self, frames_len=16, num_classes_video=400, num_classes_audio=12):
        super(DAVE, self).__init__()

        self.audio_branch = resnet18(shortcut_type='A', sample_size=64, sample_duration=frames_len, num_classes=num_classes_audio, last_fc=False, last_pool=True)
        self.video_branch = resnet18(shortcut_type='A', sample_size=112, sample_duration=frames_len, num_classes=num_classes_video, last_fc=False, last_pool=False)
        self.upscale1 = ScaleUp(512, 512)
        self.upscale2 = ScaleUp(512, 128)
        self.combinedEmbedding = nn.Conv2d(1024, 512, kernel_size=1)
        self.saliency = nn.Conv2d(128, 1, kernel_size=1)
        self._weights_init()

    def load_model(self, weights_file, device=None):
        self.load_state_dict(self._load_state_dict_(weights_file, device), strict=True)

    @staticmethod
    def _load_state_dict_(weights_file, device=None):
        if os.path.isfile(weights_file):
            # print("=> loading checkpoint '{}'".format(filepath))
            if device is None:
                checkpoint = torch.load(weights_file)
            else:
                checkpoint = torch.load(weights_file, map_location=torch.device(device))
            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            return state_dict

    def _weights_init(self):

        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)

        nn.init.kaiming_normal_(self.combinedEmbedding.weight)
        nn.init.constant_(self.combinedEmbedding.bias, 0.0)

    def forward(self, v, a=None, return_latent_streams=False):
        # V video video_frames_list of 3x16x256x320
        # A audio video_frames_list of 3x16x64x64
        # return a map of 32x40

        xV1 = self.video_branch(v)
        if a is not None:
            xA1 = self.audio_branch(a)
            xA1 = xA1.expand_as(xV1)
        else:
            # replace audio branch with zeros
            xA1 = torch.zeros_like(xV1)
        xC = torch.cat((xV1, xA1), dim=1)
        xC = torch.squeeze(xC, dim=2)
        x = self.combinedEmbedding(xC)
        x = F.relu(x, inplace=True)

        x = torch.squeeze(x, dim=2)
        x = self.upscale1(x)
        x = self.upscale2(x)
        sal = self.saliency(x)
        sal = F.relu(sal, inplace=True)
        if return_latent_streams:
            return sal, xV1, xA1
        else:
            return sal

