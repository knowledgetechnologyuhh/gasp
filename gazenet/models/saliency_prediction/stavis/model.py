from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from gazenet.models.shared_components.resnet3d import model as resnet3d
from gazenet.models.shared_components.soundnet8 import model as soundnet8

TRAINING = False


class AVModule(nn.Module):

    def __init__(self, rgb_nfilters, audio_nfilters, img_size, temp_size, hidden_layers):

        super(AVModule, self).__init__()

        self.rgb_nfilters = rgb_nfilters
        self.audio_nfilters = audio_nfilters
        self.hidden_layers = hidden_layers
        self.out_layers = 64
        self.img_size = img_size
        self.avgpool_rgb = nn.AvgPool3d((temp_size, 1, 1), stride=1)
        # Make the layers numbers equal
        self.relu = nn.ReLU()
        self.affine_rgb = nn.Linear(rgb_nfilters, hidden_layers)
        self.affine_audio = nn.Linear(audio_nfilters, hidden_layers)
        self.w_a_rgb = nn.Bilinear(hidden_layers, hidden_layers, self.out_layers, bias=True)
        self.upscale_ = nn.Upsample(scale_factor=8, mode='bilinear')


    def forward(self, rgb, audio, crop_h, crop_w):

        self.crop_w = crop_w
        self.crop_h = crop_h
        dgb = rgb[:,:,rgb.shape[2]//2-1:rgb.shape[2]//2+1,:,:]
        rgb = self.avgpool_rgb(dgb).squeeze(2)
        rgb = rgb.permute(0, 2, 3, 1)
        rgb = rgb.view(rgb.size(0), -1, self.rgb_nfilters)
        rgb = self.affine_rgb(rgb)
        rgb = self.relu(rgb)
        audio1 = self.affine_audio(audio[0].squeeze(-1).squeeze(-1))
        audio1 = self.relu(audio1)

        a_rgb_B = self.w_a_rgb(rgb.contiguous(), audio1.unsqueeze(1).expand(-1, self.img_size[0] * self.img_size[1], -1).contiguous())
        sal_bilin = a_rgb_B
        sal_bilin = sal_bilin.view(-1, self.img_size[0], self.img_size[1], self.out_layers)
        sal_bilin = sal_bilin.permute(0, 3, 1, 2)
        sal_bilin = center_crop(self.upscale_(sal_bilin), self.crop_h, self.crop_w)

        return sal_bilin

def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    # fixed indexing for PyTorch 0.4
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


class DSAMScoreDSN(nn.Module):

    def __init__(self, prev_layer, prev_nfilters, prev_nsamples):

        super(DSAMScoreDSN, self).__init__()
        i = prev_layer
        self.avgpool = nn.AvgPool3d((prev_nsamples, 1, 1), stride=1)
        # Make the layers of the preparation step
        self.side_prep = nn.Conv2d(prev_nfilters, 16, kernel_size=3, padding=1)
        # Make the layers of the score_dsn step
        self.score_dsn = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.upscale_ = nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)
        self.upscale = nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False)

    def forward(self, x, crop_h, crop_w):

        self.crop_h = crop_h
        self.crop_w = crop_w
        x = self.avgpool(x).squeeze(2)
        side_temp = self.side_prep(x)
        side = center_crop(self.upscale(side_temp), self.crop_h, self.crop_w)
        side_out_tmp = self.score_dsn(side_temp)
        side_out = center_crop(self.upscale_(side_out_tmp), self.crop_h, self.crop_w)
        return side, side_out, side_out_tmp


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def spatial_softmax(x):
    x = torch.exp(x)
    sum_batch = torch.sum(torch.sum(x, 2, keepdim=True), 3, keepdim=True)
    x_soft = torch.div(x,sum_batch)
    return x_soft

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 audiovisual=True):

        self.audiovisual = audiovisual
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64, momentum=0.1 if TRAINING else 0.0)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)

        score_dsn = nn.modules.ModuleList()
        in_channels_dsn = [64,
                           64  * block.expansion,
                           128 * block.expansion,
                           256 *block.expansion]
        temp_size_prev = [sample_duration,
                          int(sample_duration / 2),
                          int(sample_duration / 4),
                          int(sample_duration /8)]
        temp_img_size_prev = [int(sample_size / 2),
                              int(sample_size / 4),
                              int(sample_size / 8),
                              int(sample_size / 16)]
        for i in range(1,5):
            score_dsn.append(DSAMScoreDSN(i, in_channels_dsn[i-1], temp_size_prev[i-1]))
        self.score_dsn = score_dsn

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)


        if audiovisual:
            self.fuseav = nn.Conv2d(128, 1, kernel_size=1, padding=0)

            self.soundnet8 = nn.Sequential(
                soundnet8.SoundNet(momentum=0.1 if TRAINING else 0.0, reverse=True),
                nn.MaxPool2d((1, 2)))

            self.fusion3 = AVModule(in_channels_dsn[2],
                                    1024,
                                    [temp_img_size_prev[2], temp_img_size_prev[2]],
                                    temp_size_prev[3],
                                    128)

            self.fuseav.bias.data = torch.tensor([-6.0])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)
            if isinstance(m ,nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Bilinear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.fuse.bias.data = torch.tensor([-6.0])
        for i in range(0, 4):
           self.score_dsn[i].score_dsn.bias.data = torch.tensor([-6.0])


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    resnet3d.downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion, momentum=0.1 if TRAINING else 0.0))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, training=TRAINING))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, training=TRAINING))

        return nn.Sequential(*layers)

    def forward(self, x, aud):

        if self.audiovisual:
            aud = self.soundnet8(aud)
            aud = [aud]

        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        side = []
        side_out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        (tmp, tmp_, att_tmp) = self.score_dsn[0](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)
        x =torch.mul(1+att, x)
        x = self.maxpool(x)

        x = self.layer1(x)

        (tmp, tmp_, att_tmp) = self.score_dsn[1](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)
        x = torch.mul(1+att, x)
        x = self.layer2(x)

        if self.audiovisual:
            y = self.fusion3(x, aud, crop_h, crop_w)
        (tmp, tmp_, att_tmp) = self.score_dsn[2](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)
        x = torch.mul(1+att, x)
        x = self.layer3(x)

        (tmp, tmp_, att_tmp) = self.score_dsn[3](x, crop_h, crop_w)

        att = spatial_softmax(att_tmp)
        att = att.unsqueeze(1)
        side.append(tmp)
        side_out.append(tmp_)

        out = torch.cat(side[:], dim=1)

        if self.audiovisual:
            appendy = torch.cat((out, y), dim=1)
            x_out = self.fuseav(appendy)
            side_out = []
        else:
            x_out = self.fuse(out)
        side_out.append(x_out)

        x_out = {'sal': side_out}

        return x_out

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(resnet3d.Bottleneck, [3, 4, 6, 3], **kwargs)
    return model