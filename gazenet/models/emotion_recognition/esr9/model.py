#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of ESR-9 (Siqueira et al., 2020) trained on AffectNet (Mollahosseini et al., 2017) for emotion
and affect perception.
Reference:
    Siqueira, H., Magg, S. and Wermter, S., 2020. Efficient Facial Feature Learning with Wide Ensemble-based
    Convolutional Neural Networks. Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence
    (AAAI-20), pages 1–1, New York, USA.
    Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2017. AffectNet: A database for facial expression, valence,
    and arousal computing in the wild. IEEE Transactions on Affective Computing, 10(1), pp.18-31.
"""

__author__ = "Henrique Siqueira"
__email__ = "siqueira.hc@outlook.com"
__license__ = "MIT license"
__version__ = "1.0"

# Standard libraries
import os

# External libraries
import torch.nn.functional as F
import torch.nn as nn
import torch


class Base(nn.Module):
    """
        The base of the network (Ensembles with Shared Representations, ESRs) is responsible for learning low- and
        mid-level representations from the input data that are shared with an ensemble of convolutional branches
        on top of the architecture.
        In our paper (Siqueira et al., 2020), it is called shared layers or shared representations.
    """

    def __init__(self):
        super(Base, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)

        # batch-normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # convolutional, batch-normalization and pooling layers for representation learning
        x_shared_representations = F.relu(self.bn1(self.conv1(x)))
        x_shared_representations = self.pool(F.relu(self.bn2(self.conv2(x_shared_representations))))
        x_shared_representations = F.relu(self.bn3(self.conv3(x_shared_representations)))
        x_shared_representations = self.pool(F.relu(self.bn4(self.conv4(x_shared_representations))))

        return x_shared_representations


class ConvolutionalBranch(nn.Module):
    """
        Convolutional branches that compose the ensemble in ESRs. Each branch was trained on a sub-training
        set from the AffectNet dataset to learn complementary representations from the data (Siqueira et al., 2020).
        Note that, the second last layer provides eight discrete emotion labels whereas the last layer provides
        continuous values of arousal and valence levels.
    """

    def __init__(self):
        super(ConvolutionalBranch, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        # batch-normalization layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # second last, fully-connected layer related to discrete emotion labels
        self.fc = nn.Linear(512, 8)

        # last, fully-connected layer related to continuous affect levels (arousal and valence)
        self.fc_dimensional = nn.Linear(8, 2)

        # pooling layers
        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_shared_representations):
        # convolutional, batch-normalization and pooling layers
        x_conv_branch = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_conv_branch = self.pool(F.relu(self.bn2(self.conv2(x_conv_branch))))
        x_conv_branch = F.relu(self.bn3(self.conv3(x_conv_branch)))
        x_conv_branch = self.global_pool(F.relu(self.bn4(self.conv4(x_conv_branch))))
        x_conv_branch = x_conv_branch.view(-1, 512)

        # fully connected layer for emotion perception
        discrete_emotion = self.fc(x_conv_branch)

        # application of the ReLU function to neurons related to discrete emotion labels
        x_conv_branch = F.relu(discrete_emotion)

        # fully connected layer for affect perception
        continuous_affect = self.fc_dimensional(x_conv_branch)

        # returns activations of the discrete emotion output layer and arousal and valence levels
        return discrete_emotion, continuous_affect

    def forward_to_last_conv_layer(self, x_shared_representations):
        """
        Propagates activations to the last convolutional layer of the architecture.
        This method is used to generate saliency maps with the Grad-CAM algorithm (Selvaraju et al., 2017).
        Reference:
            Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017.
            Grad-cam: Visual explanations from deep networks via gradient-based localization.
            In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
        :param x_shared_representations: (ndarray) feature maps from shared layers
        :return: feature maps of the last convolutional layer
        """

        # convolutional, batch-normalization and pooling layers
        x_to_last_conv_layer = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_to_last_conv_layer = self.pool(F.relu(self.bn2(self.conv2(x_to_last_conv_layer))))
        x_to_last_conv_layer = F.relu(self.bn3(self.conv3(x_to_last_conv_layer)))
        x_to_last_conv_layer = F.relu(self.bn4(self.conv4(x_to_last_conv_layer)))

        # feature maps of the last convolutional layer
        return x_to_last_conv_layer

    def forward_from_last_conv_layer_to_output_layer(self, x_from_last_conv_layer):
        """
        Propagates activations to the second last, fully-connected layer (here referred as output layer).
        This layer represents emotion labels.
        :param x_from_last_conv_layer: (ndarray) feature maps from the last convolutional layer of this branch.
        :return: (ndarray) activations of the last second, fully-connected layer of the network
        """

        # global average polling and reshape
        x_to_output_layer = self.global_pool(x_from_last_conv_layer)
        x_to_output_layer = x_to_output_layer.view(-1, 512)

        # output layer: emotion labels
        x_to_output_layer = self.fc(x_to_output_layer)

        # returns activations of the discrete emotion output layer
        return x_to_output_layer


class ESR(nn.Module):
    """
    ESR is the unified ensemble architecture composed of two building blocks the Base and ConvolutionalBranch
    classes as described below by Siqueira et al. (2020):
    'An ESR consists of two building blocks. (1) The base (class Base) of the network is an array of convolutional
    layers for low- and middle-level feature learning. (2) These informative features are then shared with
    independent convolutional branches (class ConvolutionalBranch) that constitute the ensemble.'
    """

    def __init__(self, ensembles_num=9):
        super(ESR, self).__init__()

        # base of ESR-9 as described in the docstring (see mark 1)
        self.base = Base()

        # convolutional branches that composes ESR-9 as described in the docstring (see mark 2)
        self.convolutional_branches = []
        for i in range(1, ensembles_num + 1):
            self.convolutional_branches.append(ConvolutionalBranch())

    def load_models(self, weights_path, shared_weights_basename, ensembles_weights_basename, device=None):
        if device is None:
            self.base.load_state_dict(
                torch.load(os.path.join(weights_path, shared_weights_basename)))
        else:
            self.base.load_state_dict(
                torch.load(os.path.join(weights_path, shared_weights_basename), map_location=device))
            self.base.to(device)
        for en_idx, ensemble in enumerate(self.convolutional_branches, start=1):
            if device is None:
                ensemble.load_state_dict(torch.load(os.path.join(weights_path, ensembles_weights_basename.format(en_idx))))
            else:
                ensemble.load_state_dict(
                    torch.load(os.path.join(weights_path, ensembles_weights_basename.format(en_idx)),
                               map_location=device))
                ensemble.to(device)

    def forward(self, x):
        """
        Forward method of ESR-9.
        :param x: (ndarray) Input data.
        :return: A list of emotions and affect values from each convolutional branch in the ensemble.
        """

        # list of emotions and affect values from the ensemble
        emotions = []
        affect_values = []

        # get shared representations
        x_shared_representations = self.base(x)

        # add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch in self.convolutional_branches:
            output_emotion, output_affect = branch(x_shared_representations)
            emotions.append(output_emotion)
            affect_values.append(output_affect)

        return emotions, affect_values

    def __len__(self):
        """
        ESR with nine branches trained on AffectNet (Siqueira et al., 2020).
        :return: (int) Size of the ensemble
        """
        return len(self.convolutional_branches)
