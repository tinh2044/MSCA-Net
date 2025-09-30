import torch
from torch import nn
from transformers import TimesformerConfig, TimesformerForVideoClassification


class WrapperModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, labels):
        logits = self.backbone(x)
        return {
            "isolated_logits": logits,
            "total_loss": self.loss_fn(logits, labels.to(logits.device)),
        }


def build_i3d(num_classes, pretrained=False):
    hub_model = torch.hub.load(
        "facebookresearch/pytorchvideo", "i3d_r50", pretrained=pretrained
    )
    if (
        hasattr(hub_model, "blocks")
        and len(hub_model.blocks) >= 7
        and hasattr(hub_model.blocks[6], "proj")
    ):
        in_dim = hub_model.blocks[6].proj.in_features
        hub_model.blocks[6].proj = nn.Linear(in_dim, num_classes)
    else:
        last_block = hub_model.blocks[-1]
        if hasattr(last_block, "proj") and hasattr(last_block.proj, "in_features"):
            in_dim = last_block.proj.in_features
            last_block.proj = nn.Linear(in_dim, num_classes)
        else:
            raise AttributeError("Can not replace classification head")

    return WrapperModel(hub_model, num_classes)


class WrapperSlowfast(WrapperModel):
    def __init__(self, backbone, num_classes, alpha):
        super().__init__(backbone, num_classes)
        self.alpha = alpha

    def forward(self, x, labels):
        inputs = self.pack_inputs(x, alpha=self.alpha)
        return super().forward(inputs, labels)

    def pack_inputs(self, x, alpha=4):
        """
        Pack inputs for SlowFast model.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
            alpha: Temporal stride ratio between fast and slow pathways

        Returns:
            List of [fast_pathway, slow_pathway] tensors
        """
        assert x.ndim == 5 and x.shape[1] == 3, "Input must be (B, 3, T, H, W)"
        B, C, T, H, W = x.shape

        # Calculate target temporal dimensions based on alpha
        # For standard SlowFast: alpha=4, so if fast has 32 frames, slow has 8
        target_fast_frames = 32
        target_slow_frames = target_fast_frames // alpha

        # Fast pathway: sample target_fast_frames frames
        if T >= target_fast_frames:
            idx_fast = torch.linspace(0, T - 1, target_fast_frames).long().to(x.device)
            fast = x.index_select(dim=2, index=idx_fast)
        elif T > 1:
            # Repeat to reach target frames
            repeat_factor = (target_fast_frames + T - 1) // T
            fast = x.repeat(1, 1, repeat_factor, 1, 1)[:, :, :target_fast_frames, :, :]
        else:
            # Single frame case
            fast = x.repeat(1, 1, target_fast_frames, 1, 1)

        # Slow pathway: sample target_slow_frames frames
        if T >= target_slow_frames:
            idx_slow = torch.linspace(0, T - 1, target_slow_frames).long().to(x.device)
            slow = x.index_select(dim=2, index=idx_slow)
        elif T > 1:
            # Repeat to reach target frames
            repeat_factor = (target_slow_frames + T - 1) // T
            slow = x.repeat(1, 1, repeat_factor, 1, 1)[:, :, :target_slow_frames, :, :]
        else:
            # Single frame case
            slow = x.repeat(1, 1, target_slow_frames, 1, 1)

        # CRITICAL: Return in the correct order [slow, fast] for SlowFast model
        # The SlowFast model expects the slow pathway first, then fast pathway
        return [slow, fast]


def build_slowfast(num_classes, pretrained=False, alpha=4):
    """
    Build SlowFast model with custom number of classes.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        alpha: Temporal stride ratio (default: 4)

    Returns:
        WrapperSlowfast model
    """
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
    )

    # Replace the classification head
    if (
        hasattr(model, "blocks")
        and len(model.blocks) >= 7
        and hasattr(model.blocks[6], "proj")
    ):
        in_dim = model.blocks[6].proj.in_features
        model.blocks[6].proj = nn.Linear(in_dim, num_classes)
    else:
        last_block = model.blocks[-1]
        if hasattr(last_block, "proj") and hasattr(last_block.proj, "in_features"):
            in_dim = last_block.proj.in_features
            last_block.proj = nn.Linear(in_dim, num_classes)
        else:
            raise AttributeError("Cannot replace classification head")

    return WrapperSlowfast(model, num_classes, alpha)


class TimeSFormerWrapper(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained=False,
        attention_type="divided_space_time",
        img_size=224,
        num_frames=16,
    ):
        self.num_classes = num_classes
        super().__init__()
        self.config = TimesformerConfig(
            image_size=img_size,
            num_frames=num_frames,
            attention_type=attention_type,
            num_labels=num_classes,
        )
        self.model = TimesformerForVideoClassification(self.config)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        return {
            "isolated_logits": logits,
            "total_loss": self.loss_fn(logits, labels.to(logits.device)),
        }
