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


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with odd embedding_dim (got dim={:d})".format(
                    embedding_dim
                )
            )
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, embedding_dim, 2, dtype=torch.float)
                * -(math.log(10000.0) / embedding_dim)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embedding_dim]
        self.register_buffer("pe", pe)
        self.embedding_dim = embedding_dim

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        seq_len = inputs_embeds.size(1)
        return inputs_embeds + self.pe[:, :seq_len, :]


class AbsoluteScalarPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        device = inputs_embeds.device
        seq_len = inputs_embeds.size(1)
        positions = torch.arange(0, seq_len, device=device, dtype=inputs_embeds.dtype)
        denom = max(1, self.max_len - 1)
        norm_pos = (positions / denom).unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        abs_pe = norm_pos.expand(inputs_embeds.size(0), seq_len, self.embedding_dim)
        return inputs_embeds + abs_pe


class IdentityPositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return inputs_embeds


def build_position_embedding(config: dict) -> nn.Module:
    method = config.get("position_encoding", "learned")
    d_model = config["d_model"]
    max_pos = config.get("max_position_embeddings", 512)

    if method == "none":
        return IdentityPositionEmbedding()
    if method == "learned":
        return LearningPositionEmbedding(max_pos, d_model)
    if method == "sinecosine":
        return SinusoidalPositionEmbedding(d_model, max_pos)
    if method == "absolute":
        return AbsoluteScalarPositionEmbedding(d_model, max_pos)
    if method == "relative":
        return IdentityPositionEmbedding()
    return LearningPositionEmbedding(max_pos, d_model)


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
    def __init__(self, in_feat: int, out_feat: int, mode: str = "independent"):
        super(CoordinateMapping, self).__init__()
        self.mode = mode
        if mode == "shared":
            self.mapping = nn.Linear(in_feat, out_feat)
        else:
            self.mapping_x = nn.Linear(in_feat, out_feat)
            self.mapping_y = nn.Linear(in_feat, out_feat)

    def forward(self, x_coord: torch.Tensor, y_coord: torch.Tensor):
        if self.mode == "shared":
            x_embed = self.mapping(x_coord)
            y_embed = self.mapping(y_coord)
        else:
            x_embed = self.mapping_x(x_coord)
            y_embed = self.mapping_y(y_coord)
        return x_embed, y_embed


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_position_embeddings: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_positions = max_position_embeddings
        num_rel = 2 * max_position_embeddings - 1
        self.relative_attention_bias = nn.Embedding(num_rel, num_heads)

    def forward(self, q_len: int, k_len: int, device: torch.device):
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = context_position - memory_position  # [q_len, k_len]
        # shift to index >= 0
        rp_bucket = relative_position + (self.max_positions - 1)
        rp_bucket = rp_bucket.clamp(0, 2 * self.max_positions - 2)
        values = self.relative_attention_bias(rp_bucket)  # [q_len, k_len, num_heads]
        values = values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, q_len, k_len]
        return values
