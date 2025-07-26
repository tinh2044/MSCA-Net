from model.decoder import Decoder
from model.encoder import Encoder
from torch import nn


class SCA(nn.Module):
    def __init__(self, joint_idx, num_frame, cfg=None):
        super().__init__()
        self.joint_idx = joint_idx
        self.num_frame = num_frame

        self.x_coord_module = Encoder(cfg)
        self.y_coord_module = Decoder(cfg)

    def forward(self, x_embed, y_embed, attention_mask=None):
        x_embed = self.x_coord_module(x_embed, attention_mask)

        y_embed = self.y_coord_module(
            encoder_hidden_states=x_embed,
            encoder_attention_mask=attention_mask,
            y_embed=y_embed,
            attention_mask=attention_mask,
        )

        return y_embed
