# -*- coding: utf-8 -*-
"""This module implements the Gated Multimodal Units in PyTorch

Currently there are two versions:
Two versions, the general GMU and the simplified, bimodal unit
are described in Arevalo et al., Gated multimodal networks, 2020
(https://link.springer.com/article/10.1007/s00521-019-04559-1)

The published code of the authors contains an implementation
of the bimodal version in the Theano framework Bricks.
However, this version is a bit restrictive. It constraints
the input size with the hidden size.
See https://github.com/johnarevalo/gmu-mmimdb/blob/master/model.py

The general GMU and the bimodal version with tied gates
will be implemented here as GMU and GBU.
Now, there is also the GMU Conv2d version in here.
"""

import torch


class GMU(torch.nn.Module):
    """Gated Multimodal Unit, a hidden unit in a neural network that learns
    to combine the representation of different modalities into a single one
    via gates (similar to LSTM).

    h generally refers to the hidden state (i.e. modality information, this is
    the naming scheme chosen by the original GMU authors, but I do not like it
    that much), while
    z generally refers to the gates.
    """

    def __init__(
        self,
        in_features,
        out_features,
        modalities,
        activation=torch.tanh,
        gate_activation=torch.sigmoid,
        hidden_weight_init=lambda x: torch.nn.init.uniform_(x, -0.01, 0.01),
        gate_weight_init=lambda x: torch.nn.init.uniform_(x, -0.01, 0.01),
        gate_hidden_interaction=lambda x, y: x * y,
        gate_transformation=None,
        bias=True,
    ):
        """Init function.

        Args:
            in_features (int): vector length of a single modality
            out_features (int): number of (hidden) units / output features
            modalities (int): number of modalities
            activation (torch func): activation function for the modalities
            gate_activation (torch func): activation function for the gate
            hidden_weight_init (torch init func): init method for the neuronal
                weights
            gate_weight_init (torch init func): init method for the gate weights
            gate_hidden_interaction (lambda func): how does h and z interact
                with another. Could be linear or non-linear (e.g. x * (1+y))
            gate_transformation (lambda func): processes the gate
                activations before they interact with the hidden state,
                e.g. normalise / gain control them by
                lambda x: x / torch.sum(x, 1, keepdim=True)
            bias (bool): should the computation contain a
                bias (not specified in the original paper)

        """

        super(GMU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.modalities = modalities
        self.gates = modalities
        self.activation = activation
        self.gate_activation = gate_activation
        self.hidden_weight_init = hidden_weight_init
        self.gate_weight_init = gate_weight_init
        self.hidden_bias_init = lambda x: torch.nn.init.uniform_(x, -0.01, 0.01)
        self.gate_bias_init = lambda x: torch.nn.init.uniform_(x, -0.01, 0.01)
        self.gate_hidden_interaction = gate_hidden_interaction
        self.gate_transformation = gate_transformation
        self.W_h = self.initialize_hidden_weights()
        self.W_z = self.initialize_gate_weights()
        self.register_bias(bias)

    def register_bias(self, bias):
        """ register biases """
        if bias:
            self.hidden_bias = self.initialize_hidden_bias()
            self.gate_bias = self.initialize_gate_bias()
        else:
            self.register_parameter("hidden_bias", None)
            self.register_parameter("gate_bias", None)

    def initialize_hidden_bias(self):
        """Initializes hidden weight parameters

        Returns:
                torch.nn.Parameter
        """

        b = torch.nn.Parameter(torch.empty((1, self.modalities, self.out_features)))
        self.hidden_bias_init(b)
        return b

    def initialize_gate_bias(self):
        """Initializes hidden weight parameters

        Returns:
                torch.nn.Parameter
        """

        b = torch.nn.Parameter(torch.empty((1, self.gates, self.out_features)))
        self.gate_bias_init(b)
        return b

    def initialize_hidden_weights(self):
        """Initializes hidden weight parameters

        Returns:
                torch.nn.Parameter

        """
        # each neuron only receives the information of its associated modality
        W = torch.nn.Parameter(
            torch.empty((1, self.modalities, self.in_features, self.out_features))
        )
        return self.hidden_weight_init(W)

    def initialize_gate_weights(self):
        """Initializes gate weight parameters

        Returns:
            torch.nn.Parameter

        """
        # each gate gets the information of all modalities
        W = torch.nn.Parameter(
            torch.empty(
                (
                    self.modalities * self.in_features,
                    self.gates * self.out_features,
                )
            )
        )
        return self.gate_weight_init(W)

    @staticmethod
    def check_input(inputs):
        """Checks if the input is already a Torch tensor,
        if it is a list or tuple (hopefully one of the two),
        stack them into a tensor

        Args:
            inputs: input to the layer/cell

        Returns:
            Torch tensor of size (N,C,self.in_features)

        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.stack(inputs, 1)
        return inputs

    def get_modality_activation(self, inputs):
        """Processes the the modality information separately with a set of weights

        Args:
            inputs: input to the layer/cell

        Returns:
            Torch tensor of size (N,self.modalities,self.out_features)

        """
        h = torch.sum(inputs.unsqueeze(-1) * self.W_h, -2)
        if self.hidden_bias is not None:
            h += self.hidden_bias
        h = self.activation(h)
        return h

    def get_gate_activation(self, inputs):
        """Processes the modality information separately with a set of weights

        Args:
            inputs: input to the layer/cell

        Returns:
            Torch tensor of size (N,self.gates,self.out_features)

        """

        z = torch.matmul(inputs.view(-1, self.in_features * self.modalities), self.W_z)
        if self.gate_bias is not None:
            z = z.view(-1, self.gates, self.out_features) + self.gate_bias
        z = self.gate_activation(z)
        return z

    def forward(self, inputs):
        """Calculates the output of the unit

        Args:
            inputs (torch.Tensors): consisting of
                multiple modalities as torch.Tensors in the form NCH.
                N is batch size, C is the modalities and H the length
                of the modality vectors.

        Returns:
            A tuple of torch.Tensor of size (N, self.out_features)

        """

        inputs = self.check_input(inputs)
        h = self.get_modality_activation(inputs)
        z = self.get_gate_activation(inputs)
        if self.gate_transformation is not None:
            z = self.gate_transformation(z)
        return torch.sum(self.gate_hidden_interaction(h, z), 1), (h, z)


class GBU(GMU):
    """Gated Bimodal Unit, a hidden unit in a neural network that learns
    to combine the representation of two modalities into a single one
    via a single gate. See GMU for more general information.

    h generally refers to the hidden state, while
    z generally refers to the gate.

    Note: Since this is a specialised subclass of the GMU, most of the
        general behaviour is handled in the GMU class
    """

    def __init__(
        self,
        in_features,
        out_features,
        activation=None,
        gate_activation=None,
        hidden_weight_init=None,
        gate_weight_init=None,
        gate_hidden_interaction=None,
        gate_transformation=lambda x: torch.cat((x, (1 - x)), 1),
        bias=True,
    ):
        """Init function.

        Args:
            out_features (int): number of (hidden) units / output features
            in_features (int): vector length of a single modality
            activation (torch func, optional): activation function for the
                modalities
            gate_activation (torch func, optional): activation function for the
                gate
            hidden_weight_init (torch init func, optional): init method for the
                neuronal weights
            gate_weight_init (torch init func, optional): init method for the
                gate weights
            gate_hidden_interaction (lambda func): how does h and z interact
                with another. Could be linear or non-linear (e.g. x * (1+y))
            gate_transformation (lambda func): processes the gate
                activations before they interact with the hidden state,
                here in the bimodal case, it just concats the activations
                with the complementary probabilities

        Notes:
            # TODO hidden bias inheritance
        """

        super(GBU, self).__init__(
            in_features=in_features, out_features=out_features, modalities=2
        )
        if activation:
            self.activation = activation
        if gate_activation:
            self.gate_activation = gate_activation
        if hidden_weight_init:
            self.hidden_weight_init = hidden_weight_init
        if gate_weight_init:
            self.gate_weight_init = gate_weight_init
        if gate_hidden_interaction:
            self.gate_hidden_interaction = gate_hidden_interaction
        self.gate_transformation = gate_transformation
        self.gates = 1
        self.W_h = self.initialize_hidden_weights()
        self.W_z = self.initialize_gate_weights()
        self.register_bias(bias)


class RGMU(GMU):
    """Recurrent Gated Multimodal Unit, a hidden unit in a neural network that
    learns to combine the representation of several modalities into a single one
    incorporating recurrent activation over time. See GMU for more general
    information.

    h generally refers to the hidden state, while
    z generally refers to the gate.
    h_l are the lateral information from the hidden state, while
    z_l are the lateral information from the gate, i.e. the activations
    from the last timestep.

    Note: Since this is a specialised subclass of the GMU, most of the
        general behaviour is handled in the GMU class

    """

    def __init__(
        self,
        in_features,
        out_features,
        modalities,
        recurrent_modalities=True,
        recurrent_gates=True,
        activation=None,
        gate_activation=None,
        hidden_weight_init=None,
        lateral_hidden_weight_init=None,
        gate_weight_init=None,
        lateral_gate_weight_init=None,
        gate_hidden_interaction=None,
        gate_transformation=None,
        batch_first=True,
        bias=True,
        return_sequences=False,
    ):
        """Init function.

        Args:
            out_features (int): number of (hidden) units / output features
            in_features (int): vector length of a single modality
            modalities (int): number of modalities
            recurrent_modalities (bool, optional): if modality activation should
                incorporate recurrent information
            recurrent_gates (bool, optional): if gate activations should
                incorporate recurrent information
            activation (torch func, optional): activation function for the
                modalities
            gate_activation (torch func, optional): activation function for the
                gate
            hidden_weight_init (torch init func, optional): init method for the
                neuronal weights
            lateral_hidden_weight_init (torch init func, optional): init method
                for the recurrent neural weights
            gate_weight_init (torch init func, optional): init method for the
                gate weights
            lateral_gate_weight_init (torch init func, optional): init method
                for the recurrent gate weights
            gate_hidden_interaction (lambda func): how does h and z interact
                with another. Could be linear or non-linear (e.g. x * (1+y))
            gate_transformation (lambda func): processes the gate
                activations before they interact with the hidden state
            batch_first (bool): use batch, sequence, feature instead of
                sequence, batch, feature
                default: True
            bias (bool): tbd #todo
            return_sequences (bool): if true returns all hidden states
                from the intermediate time steps (as a list). The keras/tf
                behaviour was the inspiration for that.

        """

        super(RGMU, self).__init__(
            in_features=in_features,
            out_features=out_features,
            modalities=modalities,
        )
        if activation is not None:
            self.activation = activation
        if gate_activation is not None:
            self.gate_activation = gate_activation
        if hidden_weight_init is not None:
            self.hidden_weight_init = hidden_weight_init
        if lateral_hidden_weight_init is None:
            self.lateral_hidden_weight_init = hidden_weight_init
        else:
            self.lateral_hidden_weight_init = lateral_hidden_weight_init
        if gate_weight_init is not None:
            self.gate_weight_init = gate_weight_init
        if lateral_gate_weight_init is None:
            self.lateral_gate_weight_init = gate_weight_init
        else:
            self.lateral_gate_weight_init = lateral_gate_weight_init
        if gate_hidden_interaction is not None:
            self.gate_hidden_interaction = gate_hidden_interaction
        self.gate_transformation = gate_transformation
        self.recurrent_modalities = recurrent_modalities
        self.recurrent_gates = recurrent_gates
        self.W_h = self.initialize_hidden_weights()
        self.W_h_l = self.initialize_lateral_hidden_weights()
        self.W_z = self.initialize_gate_weights()
        self.W_z_l = self.initialize_lateral_gate_weights()
        self.batch_first = batch_first
        self.register_bias(bias)
        self.register_recurrent_bias(bias)
        self.return_sequences = return_sequences

    def register_recurrent_bias(self, bias):
        """ register recurrent biases """
        if bias:
            self.recurrent_hidden_bias = self.initialize_hidden_bias()
            self.recurrent_gate_bias = self.initialize_gate_bias()
        else:
            self.register_parameter("recurrent_hidden_bias", None)
            self.register_parameter("recurrent_gate_bias", None)

    def initialize_lateral_state(self, batch_size=1):
        """Initializes lateral state for the first forward pass

        Returns:
            a tuple of torch.Tensors

        """

        h_l = torch.zeros((batch_size, self.modalities, self.out_features), device=self.W_h.device)
        z_l = torch.zeros((batch_size, self.gates, self.out_features), device=self.W_h.device)
        return h_l, z_l

    def initialize_lateral_hidden_weights(self):
        """Initializes lateral hidden weights

        Returns:
            torch.nn.Parameter

        """
        W = torch.nn.Parameter(torch.empty((self.modalities, self.out_features)))
        if self.lateral_hidden_weight_init is not None:
            self.lateral_hiden_weight_init(W)
        return W

    def initialize_lateral_gate_weights(self):
        """Initializes lateral gate weight parameters

        Returns:
            torch.nn.Parameter

        """
        W = torch.nn.Parameter(torch.empty((self.gates, self.out_features)))
        if self.lateral_gate_weight_init is not None:
            self.lateral_gate_weight_init(W)
        return W

    def get_recurrent_modality_activation(self, inputs, h_l):
        """Processes the the modality information separately with a set of
        weights and the weighted recurrent information from the last timestep

         Args:
             inputs (Torch.Tensor): input to the layer/cell
             h_l (Torch.Tensor): activations of the last timestep

         Returns:
             Torch tensor of size (N,self.modalities,self.out_features)

        """

        h = torch.sum(inputs.unsqueeze(-1) * self.W_h, -2) + self.W_h_l * h_l
        if self.recurrent_hidden_bias is not None:
            h += self.hidden_bias + self.recurrent_hidden_bias
        return self.activation(h)

    def get_recurrent_gate_activation(self, inputs, z_l):
        """Processes the gate information separately with a set of weights
        and the weighted recurrent information from the last timestep

         Args:
            inputs (Torch.Tensor): input to the layer/cell
            z_l (Torch.Tensor): activations of the last timestep

         Returns:
             Torch tensor of size (N,self.modalities,self.out_features)

        """

        z = (
            torch.matmul(inputs.view(-1, self.in_features * self.modalities), self.W_z)
            + (self.W_z_l.unsqueeze(0) * z_l).view(-1, self.gates * self.out_features)
        ).view(-1, self.gates, self.out_features)
        if self.recurrent_gate_bias is not None:
            z += self.gate_bias + self.recurrent_gate_bias
        z = self.gate_activation(z)
        if self.gate_transformation is not None:
            z = self.gate_transformation(z)
        return z

    def step(self, inputs, lateral):
        """Calculates the output of one timestep, depending on which of the
            parts are recurrent, either modalities, gates or both

        Args:
            inputs (torch.Tensors): consisting of
                multiple modalities as torch.Tensors in the form NCH.
                N is batch size, C is the modalities and H the length
                of the modality vectors.
            lateral (tuple of torch.Tensors): tuple consisting of both,
                recurrent modality activations and recurrent gate
                activations

        Returns:
            A tuple of (torch.Tensor of size (N, self.out_features) and
                a tuple of (modality and gate activations)).
        """

        inputs = self.check_input(inputs)
        h_l, z_l = lateral
        if self.recurrent_modalities:
            h = self.get_recurrent_modality_activation(inputs, h_l)
        else:
            h = self.get_modality_activation(inputs)

        if self.recurrent_gates:
            z = self.get_recurrent_gate_activation(inputs, z_l)
        else:
            z = self.get_gate_activation(inputs)

        return torch.sum(self.gate_hidden_interaction(h, z), 1), (h, z)

    def forward(self, inputs, lateral=None):
        """Applies the layer computation to the whole sequence

        Args:
            inputs (torch.Tensors): consisting of
                multiple modalities as torch.Tensors in the form
                if batch_first: NSCH.
                N is batch size, S is sequence, C is the modalities and H the
                length
                of the modality vectors
                else: SNCH
            lateral (tuple of torch.Tensors): tuple consisting of both,
                recurrent modality activations and recurrent gate
                activations, if none is supplied, the lateral is intialized as zeros

        Returns:
            A tuple of (torch.Tensor of size (N, self.out_features) and
                a tuple of (modality (N,modalities,self.out_features) and gate
                activations (N,gates,self.out_features)).
                If return_sequences, then we follow the batch_first approach,
                where the dimensions are N, sequences, self.out_feautures.
                The lateral tuples will simply be in a list (for now).
        """
        if lateral is None:
            lateral = self.initialize_lateral_state()

        if self.return_sequences:
            output_sequences = []
            lateral_sequences = []

        if self.batch_first:
            for i in range(inputs.shape[1]):
                output, lateral = self.step(inputs[:, i], lateral)
                if self.return_sequences:
                    output_sequences.append(output)
                    lateral_sequences.append(lateral)
        else:
            for data in inputs:
                output, lateral = self.step(data, lateral)
                if self.return_sequences:
                    output_sequences.append(output)
                    lateral_sequences.append(lateral)
        if self.return_sequences:
            return torch.stack(output_sequences, 1), lateral_sequences
        else:
            return output, lateral


class GMUConv2d(torch.nn.Module):
    """Gated Multimodal Unit, a hidden unit in a neural network that learns
    to combine the representation of different modalities into a single one
    via gates (similar to LSTM).
    Here, a specialised version is used that takes as input feature maps,
    or general 2d input, convolves over these maps and subsequently,
    outputs feature maps.
    The only real difference to the non-conv versions is that the states
    and values of the units are feature maps and not scalars.

    h generally refers to the hidden state, while
    z generally refers to the gates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modalities,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        activation=torch.tanh,
        gate_activation=torch.sigmoid,
        hidden_weight_init=lambda x: torch.nn.init.uniform_(x, -0.01, 0.01),
        gate_weight_init=lambda x: torch.nn.init.uniform_(x, -0.01, 0.01),
        gate_hidden_interaction=lambda x, y: x * y,
        gate_transformation=None,
        bias=True,
    ):
        """Init function.

        Args:
            in_channels (int): number of input channels of each modality
            out_channels (int): number of (hidden) units / output feature maps
            modalities (int): number of modalities
            activation (torch func): activation function for the modalities
            gate_activation (torch func): activation function for the gate
            weight_init (torch init func): init method for the neuronal weights
            gate_weight_init (torch init func): init method for the gate weights
            gate_hidden_interaction (lambda func): how does h and z interact
                with another. Could be linear or non-linear (e.g. x * (1+y))
            gate_transformation (lambda func): processes the gate
                activations before they interact with the hidden state,
                e.g. normalise / gain control them by
                lambda x: x / torch.sum(x, 1, keepdim=True)

        Note: at the moment, the input feature maps have to be streamlined in
                the channel dimension.
                i.e. they all have to have the same number of channels.
        """

        super(GMUConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modalities = modalities
        self.gates = modalities
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.activation = activation
        self.gate_activation = gate_activation
        self.hidden_weight_init = hidden_weight_init
        self.gate_weight_init = gate_weight_init
        self.hidden_bias_init = lambda x: torch.nn.init.uniform_(
            x, -0.01, 0.01
        )  # make it as keyword?
        self.gate_bias_init = lambda x: torch.nn.init.uniform_(x, -0.01, 0.01)
        self.gate_hidden_interaction = gate_hidden_interaction
        self.gate_transformation = gate_transformation
        self.W_h = self.initialize_hidden_weights()
        self.W_z = self.initialize_gate_weights()
        self.register_bias(bias)

    def register_bias(self, bias):
        if bias:
            self.hidden_bias = self.initialize_hidden_bias()
            self.gate_bias = self.initialize_gate_bias()
        else:
            self.register_parameter("hidden_bias", None)
            self.register_parameter("gate_bias", None)

    def initialize_hidden_bias(self):
        """

        Returns:
                torch.nn.Parameter
        """
        b = torch.nn.Parameter(torch.empty((self.modalities * self.out_channels)))
        self.hidden_bias_init(b)
        return b

    def initialize_gate_bias(self):
        """

        Returns:
                torch.nn.Parameter
        """
        b = torch.nn.Parameter(torch.empty((self.gates * self.out_channels)))
        self.gate_bias_init(b)
        return b

    def initialize_gate_weights(self):
        """Initializes gate weight/kernel parameters

        Returns:
            torch.nn.Parameter

        """
        # each gate gets the information of all modalities
        # one gate per modality
        W = torch.nn.Parameter(
            torch.empty(
                (
                    self.gates * self.out_channels,
                    self.modalities * self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        )
        if self.gate_weight_init is not None:
            self.gate_weight_init(W)
        return W

    def initialize_hidden_weights(self):
        """Initializes hidden weight/kernel parameters

        Returns:
                torch.nn.Parameter

        """
        # each neuron only receives the information of its associated modality
        W = torch.nn.Parameter(
            torch.empty(
                (
                    self.modalities * self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        )
        if self.hidden_weight_init is not None:
            self.hidden_weight_init(W)
        return W

    def get_modality_activation(self, inputs):
        """Processes the modality information separately with a set of weights

        Notes:
            The groups parameter is a bit poorly documented. It works as follows: https://mc.ai/how-groups-work-in-pytorch-convolutions/

        Args:
            inputs: input feature map to the layer/cell

        Returns:
            Torch tensor of size (N,self.modalities,self.out_channels, *h, *w)
            The *height and *weight are determined by the input size and the
               use of padding,
               dilation, stride etc.

        """

        h = self.activation(
            torch.nn.functional.conv2d(
                inputs,
                self.W_h,
                self.hidden_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.modalities,
            )
        )
        return h.view(h.shape[0], self.modalities, -1, h.shape[-2], h.shape[-1])

    def get_gate_activation(self, inputs):
        """Processes the modality information with a set of weights (modalities are not treated separately but together)

        Args:
            inputs: input feature map to the layer/cell

        Returns:
            Torch tensor of size (N,self.gates,self.out_channels, *h, *w)
            The *height and *weight are determined by the input size and the
               use of padding,
               dilation, stride etc.
        """

        z = self.gate_activation(
            torch.nn.functional.conv2d(
                inputs,
                self.W_z,
                self.gate_bias,
                self.stride,
                self.padding,
                self.dilation,
                1,
            )
        )
        z = z.view(z.shape[0], self.gates, -1, z.shape[-2], z.shape[-1])
        if self.gate_transformation is not None:
            z = self.gate_transformation(z)
        return z

    def forward(self, inputs):
        """Calculates the output of the unit

        Args:
            inputs (tuple of torch.Tensors): input tuple consisting of
                multiple modalities as torch.Tensors in the form NCH.
                N is batch size, C is the modalities (as in stacked on top of each other, even if they have multiple channels each) and HW are the sizes of the feature map

        Returns:
            torch.Tensor of size (N, out_channels, *h, *w)
            The *height and *weight are determined by the input size and the
                use of padding,
                dilation, stride etc.

        """

        inputs = GMU.check_input(inputs)
        h = self.get_modality_activation(inputs)
        z = self.get_gate_activation(inputs)
        return torch.sum(self.gate_hidden_interaction(h, z), 1), (h, z)


class GBUConv2d(GMUConv2d):
    """Gated Multimodal Unit, a hidden unit in a neural network that learns
    to combine the representation of different modalities into a single one
    via gates (similar to LSTM).
    Here, a specialised version is used that takes as input feature maps,
    or general 2d input, convolves over these maps and subsequently,
    outputs feature maps.
    The only real difference to the non-conv versions is that the states
    and values of the units are feature maps and not scalars.
    GBU here indicates that only two modalities are possible for input
    and only one gate is used.

    h generally refers to the hidden state, while
    z generally refers to the gates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
        padding=None,
        dilation=None,
        activation=None,
        gate_activation=None,
        hidden_weight_init=None,
        gate_weight_init=None,
        gate_hidden_interaction=None,
        gate_transformation=lambda x: torch.cat((x, (1 - x)), 1),
        bias=True,
    ):
        """Init function.

        Args:
            in_channels (int): number of input channels of each modality
            out_channels (int): number of (hidden) units / output feature maps
            modalities (int): number of modalities
            activation (torch func): activation function for the modalities
            gate_activation (torch func): activation function for the gate
            weight_init (torch init func): init method for the neuronal weights
            gate_weight_init (torch init func): init method for the gate weights
            gate_hidden_interaction (lambda func): how does h and z interact
                with another. Could be linear or non-linear (e.g. x * (1+y))
            gate_transformation (lambda func): processes the gate
                activations before they interact with the hidden state,
                here in the bimodal case, it just concats the activations
                with the complementary probabilites


        Note: at the moment, the input feature maps have to be streamlined in
            the channel dimension.
                i.e. they all have to have the same number of channels.
        """

        super(GBUConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modalities=2,
            kernel_size=kernel_size,
        )
        if stride is not None:
            self.stride = stride
        if padding is not None:
            self.padding = padding
        if dilation is not None:
            self.dilation = dilation
        if activation is not None:
            self.activation = activation
        if gate_activation is not None:
            self.gate_activation = gate_activation
        if hidden_weight_init is not None:
            self.hidden_weight_init = hidden_weight_init
        if gate_weight_init is not None:
            self.gate_weight_init = gate_weight_init
        if gate_hidden_interaction is not None:
            self.gate_hidden_interaction = gate_hidden_interaction
        if gate_transformation is not None:
            self.gate_transformation = gate_transformation
        self.gates = 1
        self.W_h = self.initialize_hidden_weights()
        self.W_z = self.initialize_gate_weights()
        self.register_bias(bias)


class RGMUConv2d(GMUConv2d):
    """Recurrent Gated Multimodal Unit, a hidden unit in a neural network that
    learns to combine the representation of different modalities into a
    single one via gates (similar to LSTM).
    Here, a specialised version is used that takes as input feature maps,
    or general 2d input, convolves over these maps and subsequently,
    outputs feature maps.
    The only real difference to the non-conv versions is that the states
    and values of the units are feature maps and not scalars.
    Recurrent means that the either the gates or the modalities, or both,
    incorporate information from prior timesteps in there processing.

    h generally refers to the hidden state, while
    z generally refers to the gates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        modalities,
        input_size,
        recurrent_modalities=True,
        recurrent_gates=True,
        stride=None,
        padding=None,
        dilation=None,
        activation=None,
        gate_activation=None,
        hidden_weight_init=None,
        lateral_hidden_weight_init=None,
        gate_weight_init=None,
        lateral_gate_weight_init=None,
        gate_hidden_interaction=None,
        gate_transformation=None,
        batch_first=True,
        return_sequences=False,
        bias=True,
        device="cuda:0",
    ):
        """Init function.

        Args:
            in_channels (int): number of input channels of each modality
            out_channels (int): number of (hidden) units / output feature maps
            modalities (int): number of modalities
            input_size (list or tuple): height and width of the input
            recurrent_modalities (bool, optional): if modality activation should
                incorporate recurrent information
            recurrent_gates (bool, optional): if gate activations should
                incorporate recurrent information
            activation (torch func): activation function for the modalities
            gate_activation (torch func): activation function for the gate
            weight_init (torch init func): init method for the neuronal weights
            lateral_hidden_weight_init (torch init func, optional): init method
                for the recurrent neural weights
            gate_weight_init (torch init func): init method for the gate weights
            lateral_gate_weight_init (torch init func, optional): init method
                for the recurrent gate weights
            gate_hidden_interaction (lambda func): how does h and z interact
                with another. Could be linear or non-linear (e.g. x * (1+y))
            gate_transformation (lambda func): processes the gate
                activations before they interact with the hidden state,
                here in the bimodal case, it just concats the activations
                with the complementary probabilites
            device (string): gpu or cpu device


        Note: at the moment, the input feature maps have to be streamlined in
            the channel dimension.
                i.e. they all have to have the same number of channels.
        """

        super(RGMUConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            modalities=modalities,
            kernel_size=kernel_size,
        )
        self.device = device
        self.height = input_size[0]
        self.width = input_size[1]
        if stride is not None:
            self.stride = stride
        if padding is not None:
            self.padding = padding
        if dilation is not None:
            self.dilation = dilation
        if activation is not None:
            self.activation = activation
        if gate_activation is not None:
            self.gate_activation = gate_activation
        if hidden_weight_init is not None:
            self.hidden_weight_init = hidden_weight_init
        if lateral_hidden_weight_init is not None:
            self.lateral_hidden_weight_init = lateral_hidden_weight_init
        else:
            self.lateral_hidden_weight_init = self.hidden_weight_init
        if gate_weight_init is not None:
            self.gate_weight_init = gate_weight_init
        if lateral_gate_weight_init is not None:
            self.lateral_gate_weight_init = lateral_gate_weight_init
        else:
            self.lateral_gate_weight_init = self.gate_weight_init
        if gate_hidden_interaction is not None:
            self.gate_hidden_interaction = gate_hidden_interaction
        if gate_transformation is not None:
            self.gate_transformation = gate_transformation
        self.gates = modalities
        self.recurrent_modalities = recurrent_modalities
        self.recurrent_gates = recurrent_gates
        self.return_sequences = return_sequences
        self.batch_first = batch_first
        self.W_h = self.initialize_hidden_weights()
        self.W_h_l = self.initialize_lateral_hidden_weights()
        self.W_z = self.initialize_gate_weights()
        self.W_z_l = self.initialize_lateral_gate_weights()
        self.register_bias(bias)
        self.register_recurrent_bias(bias)

    def register_recurrent_bias(self, bias):
        if bias:
            self.recurrent_hidden_bias = self.initialize_hidden_bias()
            self.recurrent_gate_bias = self.initialize_gate_bias()
        else:
            self.register_parameter("recurrent_hidden_bias", None)
            self.register_parameter("recurrent_gate_bias", None)

    def initialize_lateral_state(self, batch_size=1):
        """ Todo: Docstring"""
        h_l = torch.zeros(
            (
                batch_size,
                self.modalities * self.out_channels,
                (self.height - self.kernel_size + self.padding * 2) // self.stride + 1,
                (self.width - self.kernel_size + self.padding * 2) // self.stride + 1,
            )
        )
        z_l = torch.zeros(
            (
                batch_size,
                self.gates * self.out_channels,
                self.height - (self.kernel_size - 1) + self.padding * 2,
                self.width - (self.kernel_size - 1) + self.padding * 2,
            )
        )
        return h_l.to(self.device), z_l.to(self.device)

    def initialize_lateral_gate_weights(self):
        """Initializes gate weight/kernel parameters

        Returns:
            torch.nn.Parameter

        """
        # the recurrent processing takes as input the output of the gate
        # processing, i.e. one feature map per gate, per RGMUCell
        W = torch.nn.Parameter(
            torch.empty(
                (
                    self.gates * self.out_channels,
                    1,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        )
        if self.gate_weight_init is not None:
            self.gate_weight_init(W)
        return W

    def initialize_lateral_hidden_weights(self):
        """Initializes hidden weight/kernel parameters

        Returns:
                torch.nn.Parameter

        """
        # as input we receive the output of the modality processing,
        # i.e. one feature map per modality, per RGMUCell
        W = torch.nn.Parameter(
            torch.empty(
                (
                    self.modalities * self.out_channels,
                    1,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        )
        if self.hidden_weight_init is not None:
            self.hidden_weight_init(W)
        return W

    def get_recurrent_modality_activation(self, inputs, h_l):
        """Processes the the modality information separately with a set of
        weights and the weighted recurrent information from the last timestep

         Args:
             inputs (Torch.Tensor): input to the layer/cell
             h_l (Torch.Tensor): activations of the last timestep

         Returns:
             Torch tensor of size (N,self.modalities,self.out_channels, *h, *w)

        """

        h = self.activation(
            torch.nn.functional.conv2d(
                inputs,
                self.W_h,
                self.hidden_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.modalities,
            )
            + torch.nn.functional.conv2d(
                h_l,
                self.W_h_l,
                self.recurrent_hidden_bias,
                self.stride,
                (self.kernel_size - 1) // 2,
                self.dilation,
                self.modalities * self.out_channels,
            )
        )
        return h.view(h.shape[0], self.modalities, -1, h.shape[-2], h.shape[-1])

    def get_recurrent_gate_activation(self, inputs, z_l):
        """Processes the gate information separately with a set of weights
        and the weighted recurrent information from the last timestep

         Args:
            inputs (Torch.Tensor): input to the layer/cell
            z_l (Torch.Tensor): activations of the last timestep

         Returns:
             Torch tensor of size (N,self.gates,self.out_channels, *h, *w)

        """

        z = self.gate_activation(
            torch.nn.functional.conv2d(
                inputs,
                self.W_z,
                self.gate_bias,
                self.stride,
                self.padding,
                self.dilation,
                1,
            )
            + torch.nn.functional.conv2d(
                z_l,
                self.W_z_l,
                self.recurrent_gate_bias,
                self.stride,
                (self.kernel_size - 1) // 2,
                self.dilation,
                self.gates * self.out_channels,
            )
        )
        z = z.view(z.shape[0], self.gates, -1, z.shape[-2], z.shape[-1])
        if self.gate_transformation is not None:
            z = self.gate_transformation(z)
        return z

    def step(self, inputs, lateral):
        """ Copy from RGMU but adapt to 2D"""

        h_l, z_l = lateral
        if self.recurrent_modalities:
            h = self.get_recurrent_modality_activation(inputs, h_l)
        else:
            h = self.get_modality_activation(inputs)

        if self.recurrent_gates:
            z = self.get_recurrent_gate_activation(inputs, z_l)
        else:
            z = self.get_gate_activation(inputs)

        return torch.sum(self.gate_hidden_interaction(h, z), 1), (
            h.view(
                h.shape[0],
                self.modalities * self.out_channels,
                h.shape[-2],
                h.shape[-1],
            ),
            z.view(
                z.shape[0],
                self.gates * self.out_channels,
                z.shape[-2],
                z.shape[-1],
            ),
        )

    def forward(self, inputs, lateral=None):
        """# TODO adapt docstring

        Args:
            inputs (torch.Tensors): consisting of
                multiple modalities as torch.Tensors in the form
                if batch_first: NSCH.
                N is batch size, S is sequence, C is the modalities and H the
                length
                of the modality vectors
                else: SNCH
            lateral (tuple of torch.Tensors): tuple consisting of both,
                recurrent modality activations and recurrent gate
                activations, if none is supplied, the lateral is intialized as zeros

        Returns:
            A tuple of (torch.Tensor of size (N, self.out_features) and
                a tuple of (modality (N,modalities,self.out_features) and gate
                activations (N,gates,self.out_features)).
                If return_sequences, then we follow the batch_first approach,
                where the dimensions are N, sequences, self.out_feautures.
                The lateral tuples will simply be in a list (for now).
        """
        if lateral is None:
            lateral = self.initialize_lateral_state()
            # So if you run the code on GPU, it leads to errors.

        if self.return_sequences:
            output_sequences = []
            lateral_sequences = []

        if self.batch_first:
            for i in range(inputs.shape[1]):
                output, lateral = self.step(inputs[:, i], lateral)
                if self.return_sequences:
                    output_sequences.append(output)
                    lateral_sequences.append(lateral)
        else:
            for data in inputs:
                output, lateral = self.step(data, lateral)
                if self.return_sequences:
                    output_sequences.append(output)
                    lateral_sequences.append(lateral)
        if self.return_sequences:
            return torch.stack(output_sequences, 1), lateral_sequences
        else:
            return output, lateral


if __name__ == "__main__":
    # import numpy as np

    # np.random.seed(1337)
    # from torch.utils.data import Dataset
    # import multimodal as mm

    rgmuconv_in1_out2_mod3 = RGMUConv2d(
        in_channels=1,
        out_channels=2,
        modalities=3,
        kernel_size=3,
        input_size=[5, 5],
    )

    input2d_5x5_c1_mod3_len4 = torch.ones((8, 4, 3, 5, 5))
    lat = rgmuconv_in1_out2_mod3.initialize_lateral_state()
    h, z = lat

    rgmuconv_in1_out2_mod3(input2d_5x5_c1_mod3_len4, lat)
