from torch import nn
from model.decoder import Decoder
from model.encoder import Encoder
from model.layers import CoordinateMapping

from model.residual import ResidualNetwork


class KeypointModule(nn.Module):
    def __init__(self, joint_idx, num_frame, cfg=None):
        super().__init__()
        self.joint_idx = joint_idx
        self.num_frame = num_frame
        self.coordinate_mapping = CoordinateMapping(len(joint_idx), cfg["d_model"])
        self.sca = SCA(joint_idx, num_frame, cfg)
        self.residual = ResidualNetwork(cfg["residual_blocks"])

    def forward(self, keypoints, attention_mask=None, return_attn_map=False):
        x = keypoints[:, :, :, 0]
        y = keypoints[:, :, :, 1]

        x_embed, y_embed = self.coordinate_mapping(x, y)

        sca_output = self.sca(
            x_embed, y_embed, attention_mask, return_attn_map=return_attn_map
        )
        outputs = sca_output["outputs"]
        x_attn_map = sca_output["x_attn_map"]
        y_attn_map = sca_output["y_attn_map"]
        cross_attn_map = sca_output["cross_attn_map"]

        outputs, _ = self.residual(outputs)

        return outputs, {
            "self_attn_maps": x_attn_map,
            "causal_attn_maps": y_attn_map,
            "cross_attn_maps": cross_attn_map,
        }


class SCA(nn.Module):
    def __init__(self, joint_idx, num_frame, cfg=None):
        super().__init__()
        self.joint_idx = joint_idx
        self.num_frame = num_frame

        self.x_coord_module = Encoder(cfg)
        self.y_coord_module = Decoder(cfg)

    def forward(self, x_embed, y_embed, attention_mask=None, return_attn_map=False):
        x_embed, x_attn_map = self.x_coord_module(
            x_embed, attention_mask, return_attn_map=return_attn_map
        )

        outputs, y_attn_map, cross_attn_map = self.y_coord_module(
            encoder_hidden_states=x_embed,
            encoder_attention_mask=attention_mask,
            y_embed=y_embed,
            attention_mask=attention_mask,
            return_attn_map=return_attn_map,
        )

        return {
            "outputs": outputs,
            "x_attn_map": x_attn_map,
            "y_attn_map": y_attn_map,
            "cross_attn_map": cross_attn_map,
        }
