import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init, normal_init


nb_timestep = 4


# https://github.com/PanoAsh/Saliency-Attentive-Model-Pytorch/blob/master/main.py
class AttentiveLSTM(nn.Module):

    def __init__(self, nb_features_in, nb_features_out, nb_features_att, nb_rows, nb_cols):
        super(AttentiveLSTM, self).__init__()

        # define the fundamantal parameters
        self.nb_features_in = nb_features_in
        self.nb_features_out = nb_features_out
        self.nb_features_att = nb_features_att
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

        # define convs
        self.W_a = nn.Conv2d(in_channels=self.nb_features_att, out_channels=self.nb_features_att,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_a = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_att,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.V_a = nn.Conv2d(in_channels=self.nb_features_att, out_channels=1,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=False)

        self.W_i = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_i = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_f = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_f = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_c = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_c = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_o = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_o = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        # define activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # define number of temporal steps
        self.nb_ts = nb_timestep

    def forward(self, x):
        # get the current cell memory and hidden state
        h_curr, c_curr = x, x

        for i in range(self.nb_ts):

            # the attentive model
            my_Z = self.V_a(self.tanh(self.W_a(h_curr) + self.U_a(x)))
            my_A = self.softmax(my_Z)
            AM_cL = my_A * x

            # the convLSTM model
            my_I = self.sigmoid(self.W_i(AM_cL) + self.U_i(h_curr))
            my_F = self.sigmoid(self.W_f(AM_cL) + self.U_f(h_curr))
            my_O = self.sigmoid(self.W_o(AM_cL) + self.U_o(h_curr))
            my_G = self.tanh(self.W_c(AM_cL) + self.U_c(h_curr))
            c_next = my_G * my_I +  my_F * c_curr
            h_next = self.tanh(c_next) * my_O

            c_curr = c_next
            h_curr = h_next

        return h_curr


class SequenceAttentiveLSTM(AttentiveLSTM):
    def __init__(self, *args, sequence_len=2, sequence_norm=True, **kwargs):
        super().__init__(*args, **kwargs)

        if sequence_norm:
            self.sequence_norm = nn.BatchNorm3d(sequence_len)
            # self.sequence_len = sequence_len
        else:
            self.sequence_norm = lambda x : x
            # self.sequence_len = None

    def forward(self, x):
        x = self.sequence_norm(x)
        # get the current cell memory and hidden state
        h_curr, c_curr = x[:,0], x[:,0]

        for i in range(x.shape[1]):  # for i in range(self.sequence_len):
            # the attentive model
            my_Z = self.V_a(self.tanh(self.W_a(h_curr) + self.U_a(x[:,i])))
            my_A = self.softmax(my_Z)
            AM_cL = my_A * x[:,i]

            # the convLSTM model
            my_I = self.sigmoid(self.W_i(AM_cL) + self.U_i(h_curr))
            my_F = self.sigmoid(self.W_f(AM_cL) + self.U_f(h_curr))
            my_O = self.sigmoid(self.W_o(AM_cL) + self.U_o(h_curr))
            my_G = self.tanh(self.W_c(AM_cL) + self.U_c(h_curr))
            c_next = my_G * my_I + my_F * c_curr
            h_next = self.tanh(c_next) * my_O

            c_curr = c_next
            h_curr = h_next

        return h_curr


# https://github.com/chenxy99/Scanpaths/blob/main/COCO_Search18/models/baseline_attention_multihead.py
class SpatialSemanticLSTM(nn.Module):
    def __init__(self, embed_size=512):
        super(SpatialSemanticLSTM, self).__init__()
        #LSTM gates
        self.input_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.memory_x = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.input_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.memory_h = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.input = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.forget = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)
        self.output = nn.Conv2d(embed_size, embed_size, kernel_size=3, padding=1, stride=1, bias=True)

        self.init_weights()

    def forward(self, x, state, spatial, semantic):
        batch, channel, col, row = x.size()

        spatial_semantic = spatial.unsqueeze(1) * semantic.unsqueeze(-1).unsqueeze(-1)

        h, c = state[0], state[1]
        i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input(spatial_semantic))
        f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget(spatial_semantic))
        o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output(spatial_semantic))
        g = torch.tanh(self.memory_x(x) + self.memory_h(h))

        next_c = f * c + i * g
        h = o * next_c
        state = (h, next_c)

        return h, state

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
