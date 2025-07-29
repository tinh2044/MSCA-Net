import torch
from torch import nn
import math
from torch.nn import functional as F


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class LearningPositionEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, inputs_embeds):
        bsz, seq_len = inputs_embeds.shape[:2]

        positions = (
            torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
            .expand(bsz, -1)
            .to(inputs_embeds.device)
        )
        positions_embeddings = super().forward(positions + self.offset)

        return inputs_embeds + positions_embeddings


class StaticPositionalEncoding(nn.Module):
    def __init__(self, size: int = 0, max_len: int = 5000):
        super(StaticPositionalEncoding, self).__init__()
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        return emb + self.pe[:, : emb.size(1)]


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, in_dim)

        self.dropout = dropout

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CoordinateMapping(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(CoordinateMapping, self).__init__()

        self.mapping_x = nn.Linear(in_feat, out_feat)
        self.mapping_y = nn.Linear(in_feat, out_feat)

    def forward(self, x_coord, y_coord):
        x_embed = self.mapping_x(x_coord)

        y_embed = self.mapping_y(y_coord)

        return x_embed, y_embed
