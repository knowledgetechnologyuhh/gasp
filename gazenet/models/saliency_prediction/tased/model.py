"""
code from: https://raw.githubusercontent.com/MichiganCOG/TASED-Net/master/model.py
"""

import torch
from torch import nn

from gazenet.models.shared_components.conv3d import model as conv3d


class TASED_v2(nn.Module):
    def __init__(self):
        super(TASED_v2, self).__init__()
        self.base1 = nn.Sequential(
            conv3d.SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            conv3d.BasicConv3d(64, 64, kernel_size=1, stride=1),
            conv3d.SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxp2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.maxm2 = nn.MaxPool3d(kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0))
        self.maxt2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True)
        self.base2 = nn.Sequential(
            conv3d.Mixed_3b(),
            conv3d.Mixed_3c(),
        )
        self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.maxm3 = nn.MaxPool3d(kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0))
        self.maxt3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True)
        self.base3 = nn.Sequential(
            conv3d.Mixed_4b(),
            conv3d.Mixed_4c(),
            conv3d.Mixed_4d(),
            conv3d.Mixed_4e(),
            conv3d.Mixed_4f(),
        )
        self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        self.base4 = nn.Sequential(
            conv3d.Mixed_5b(),
            conv3d.Mixed_5c(),
        )
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1024, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.unpool1 = nn.MaxUnpool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
        self.convtsp2 = nn.Sequential(
            nn.ConvTranspose3d(832, 480, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.unpool2 = nn.MaxUnpool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convtsp3 = nn.Sequential(
            nn.ConvTranspose3d(480, 192, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.unpool3 = nn.MaxUnpool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convtsp4 = nn.Sequential(
            nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(4, 4, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(4, 4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y3 = self.base1(x)
        y = self.maxp2(y3)
        y3 = self.maxm2(y3)
        _, i2 = self.maxt2(y3)
        y2 = self.base2(y)
        y = self.maxp3(y2)
        y2 = self.maxm3(y2)
        _, i1 = self.maxt3(y2)
        y1 = self.base3(y)
        y = self.maxt4(y1)
        y, i0 = self.maxp4(y)
        y0 = self.base4(y)

        z = self.convtsp1(y0)
        z = self.unpool1(z, i0)
        z = self.convtsp2(z)
        z = self.unpool2(z, i1, y2.size())
        z = self.convtsp3(z)
        z = self.unpool3(z, i2, y3.size())
        z = self.convtsp4(z)
        z = z.view(z.size(0), z.size(3), z.size(4))

        return z



