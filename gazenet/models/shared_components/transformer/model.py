import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, feat_size, dropout=0.1, max_len=4):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, feat_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feat_size, 2).float() * (-math.log(10000.0) / feat_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe
        # return self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, feat_size, hidden_size=256, nhead=4, num_encoder_layers=3, max_len=4, num_decoder_layers=-1,
                 num_queries=4, spatial_dim=-1):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feat_size, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(feat_size, nhead, hidden_size)

        self.spatial_dim = spatial_dim
        if self.spatial_dim != -1:
            transformer_encoder_spatial_layers = nn.TransformerEncoderLayer(spatial_dim, nhead, hidden_size)
            self.transformer_encoder_spatial = nn.TransformerEncoder(transformer_encoder_spatial_layers,
                                                                     num_encoder_layers)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.use_decoder = (num_decoder_layers != -1)

        if self.use_decoder:
            decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead, hidden_size)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers,
                                                             norm=nn.LayerNorm(hidden_size))
            self.tgt_pos = nn.Embedding(num_queries, hidden_size).weight
            assert self.tgt_pos.requires_grad == True

    def forward(self, embeddings, idx):
        ''' embeddings: CxBxCh*H*W '''
        # print(embeddings.shape)
        batch_size = embeddings.size(1)

        if self.spatial_dim != -1:
            embeddings = embeddings.permute((2, 1, 0))
            embeddings = self.transformer_encoder_spatial(embeddings)
            embeddings = embeddings.permute((2, 1, 0))

        x = self.pos_encoder(embeddings)
        x = self.transformer_encoder(x)
        if self.use_decoder:
            if idx != -1:
                tgt_pos = self.tgt_pos[idx].unsqueeze(0)
                # print(tgt_pos.size())
                tgt_pos = tgt_pos.unsqueeze(1).repeat(1, batch_size, 1)
            else:
                tgt_pos = self.tgt_pos.unsqueeze(1).repeat(1, batch_size, 1)
            tgt = torch.zeros_like(tgt_pos)
            x = self.transformer_decoder(tgt + tgt_pos, x)
        return x
