import torch
import torch.nn as nn


class SoundNet(nn.Module):

    def __init__(self, momentum=0.1, reverse=False):
        super(SoundNet, self).__init__()
        self.reverse = reverse

        self.conv1 = nn.Conv2d(1, 16, kernel_size=self._rev(64, 1), stride=self._rev(2, 1),
                               padding=self._rev(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=momentum)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(self._rev(8, 1), stride=self._rev(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=self._rev(32, 1), stride=self._rev(2, 1),
                               padding=self._rev(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=momentum)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d(self._rev(8, 1), stride=self._rev(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=self._rev(16, 1), stride=self._rev(2, 1),
                               padding=self._rev(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=momentum)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=self._rev(8, 1), stride=self._rev(2, 1),
                               padding=self._rev(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=momentum)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=self._rev(4, 1), stride=self._rev(2, 1),
                               padding=self._rev(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=momentum)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d(self._rev(4, 1), stride=self._rev(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=self._rev(4, 1), stride=self._rev(2, 1),
                               padding=self._rev(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=momentum)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=self._rev(4, 1), stride=self._rev(2, 1),
                               padding=self._rev(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=momentum)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
                                    stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)

        return x

    def _rev(self, *tup):
        if self.reverse:
            new_tup = tup[::-1]
            return new_tup
        else:
            return tup
