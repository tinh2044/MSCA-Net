import torch
import torch.nn as nn
from .attention import SelfAttention
from .layers import LearningPositionEmbedding
from .utils import create_attention_mask


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]

        self.self_attn = SelfAttention(
            d_model=self.d_model,
            num_heads=config["encoder_attention_heads"],
            dropout=config["attention_dropout"],
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = config["dropout"]
        self.activation_fn = nn.GELU()
        self.activation_dropout = config["activation_dropout"]
        self.fc1 = nn.Linear(self.d_model, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, attention_mask, return_attn_map=False):
        residual = hidden_states
        hidden_states, attn_map = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_attn_map=return_attn_map,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        return hidden_states, attn_map


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = config["dropout"]
        self.layerdrop = config["encoder_layerdrop"]

        embed_dim = config["d_model"]
        self.embed_positions = LearningPositionEmbedding(
            config["max_position_embeddings"],
            embed_dim,
        )

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config["encoder_layers"])]
        )

        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(self, x_embed, attention_mask, return_attn_map=False):
        x_embed = self.embed_positions(x_embed)

        hidden_states = self.layernorm_embedding(x_embed)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.layerdrop, training=self.training
        )

        attention_mask = create_attention_mask(attention_mask, x_embed.dtype)

        attn_maps = []

        for encoder_layer in self.layers:
            hidden_states, attn_map = encoder_layer(
                hidden_states, attention_mask, return_attn_map=return_attn_map
            )
            if return_attn_map:
                attn_maps.append(attn_map)


        return hidden_states, attn_maps[-1] if return_attn_map else None
