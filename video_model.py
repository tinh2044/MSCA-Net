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


def build_i3d(num_classes, pretrained=True):
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
        assert x.ndim == 5 and x.shape[1] == 3, "Input must be (B, 3, T, H, W)"
        B, C, T, H, W = x.shape
        fast = x
        idx = torch.linspace(0, T - 1, max(T // alpha, 1)).long().to(x.device)
        slow = x.index_select(dim=2, index=idx)
        return [fast, slow]


def build_slowfast(num_classes, pretrained=True, alpha=4):
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
    )

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
            raise AttributeError("Can not replace classification head")

    return WrapperSlowfast(model, num_classes, alpha)


class TimeSFormerWrapper(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained=True,
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
        if pretrained:
            self.model = TimesformerForVideoClassification.from_pretrained(
                "facebook/timesformer-base-finetuned-k400"
            )
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, num_classes
            )
        else:
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
