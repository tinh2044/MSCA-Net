import torch
from torch import nn
from model.keypoint_module import KeypointModule
from model.fusion import CoordinatesFusion
from model.loss import SeqKD
import torch.nn.functional as F


class RecognitionHead(nn.Module):
    def __init__(self, cfg, num_classes=None):
        super().__init__()

        self.left_gloss_classifier = nn.Linear(cfg["residual_blocks"][-1], num_classes)
        self.right_gloss_classifier = nn.Linear(cfg["residual_blocks"][-1], num_classes)
        self.body_gloss_classifier = nn.Linear(cfg["residual_blocks"][-1], num_classes)
        self.fuse_coord_classifier = nn.Linear(cfg["out_fusion_dim"], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        left_output,
        right_output,
        fuse_output,
        body_output,
    ):
        left_logits = self.left_gloss_classifier(left_output)
        right_logits = self.right_gloss_classifier(right_output)
        body_logits = self.body_gloss_classifier(body_output)
        gloss_logits = self.fuse_coord_classifier(fuse_output)

        outputs = {
            "left": left_logits,
            "right": right_logits,
            "body": body_logits,
            "gloss_logits": gloss_logits,
        }
        return outputs


class MSCA_Net(torch.nn.Module):
    def __init__(self, cfg, device="cpu", num_classes=None):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.num_classes = num_classes
        self.cfg["fuse_idx"] = cfg["left_idx"] + cfg["right_idx"] + cfg["body_idx"]
        self.left_idx = cfg["left_idx"]

        self.task = cfg.get("task", "continuous")

        self.islr_pooling = cfg.get("islr_pooling", "mean")
        self.islr_head = cfg.get("islr_head", "fuse")

        self.self_distillation = cfg["self_distillation"]
        self.distillation_on_islr = cfg.get("distillation_on_islr", True)

        self.gradient_clip_val = cfg.get("gradient_clip_val", 1.0)

        self.body_encoder = KeypointModule(
            cfg["body_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.left_encoder = KeypointModule(
            cfg["left_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.right_encoder = KeypointModule(
            cfg["right_idx"], num_frame=cfg["num_frame"], cfg=cfg
        )
        self.coordinates_fusion = CoordinatesFusion(
            cfg["in_fusion_dim"], cfg["out_fusion_dim"], 0.2
        )
        self.recognition_head = RecognitionHead(cfg, num_classes=num_classes)

        self.loss_fn = nn.CTCLoss(reduction="none", zero_infinity=True, blank=0)
        self.distillation_loss = SeqKD()

        self.isolated_ce = nn.CrossEntropyLoss(reduction="mean")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, src_input, return_attention_maps=False, **kwargs):
        if torch.cuda.is_available() and self.device == "cuda":
            src_input = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in src_input.items()
            }

        keypoints = src_input["keypoints"]
        mask = src_input["mask"]

        if torch.isnan(keypoints).any() or torch.isinf(keypoints).any():
            raise ValueError("NaN or inf in input keypoints")

        attention_data = {}

        body_embed, body_attn_maps = self.body_encoder(
            keypoints[:, :, self.cfg["body_idx"], :],
            mask,
            return_attn_map=return_attention_maps,
        )
        left_embed, left_attn_maps = self.left_encoder(
            keypoints[:, :, self.cfg["left_idx"], :],
            mask,
            return_attn_map=return_attention_maps,
        )
        right_embed, right_attn_maps = self.right_encoder(
            keypoints[:, :, self.cfg["right_idx"], :],
            mask,
            return_attn_map=return_attention_maps,
        )

        attention_data = {
            "body_attention_data": body_attn_maps,
            "left_attention_data": left_attn_maps,
            "right_attention_data": right_attn_maps,
        }

        if torch.isnan(body_embed).any() or torch.isinf(body_embed).any():
            raise ValueError("NaN or inf in body_embed")
        if torch.isnan(left_embed).any() or torch.isinf(left_embed).any():
            raise ValueError("NaN or inf in left_embed")
        if torch.isnan(right_embed).any() or torch.isinf(right_embed).any():
            raise ValueError("NaN or inf in right_embed")

        if return_attention_maps:
            fuse_embed, fusion_attn_data = self.coordinates_fusion(
                left_embed, right_embed, body_embed, return_attn_map=True
            )
        else:
            fuse_embed = self.coordinates_fusion(
                left_embed, right_embed, body_embed, return_attn_map=False
            )

        if torch.isnan(fuse_embed).any() or torch.isinf(fuse_embed).any():
            raise ValueError("NaN or inf in fuse_embed")

        if self.task == "isolated":
            head_outputs = self.recognition_head(
                left_embed, right_embed, fuse_embed, body_embed
            )

            for k, v in head_outputs.items():
                if torch.isnan(v).any():
                    raise ValueError(f"NaN in {k}")
                if torch.isinf(v).any():
                    raise ValueError(f"inf in {k}")

            def pool_time(x):
                if self.islr_pooling == "max":
                    return torch.amax(x, dim=1)
                return torch.mean(x, dim=1)

            if self.islr_head == "left":
                isolated_logits = pool_time(head_outputs["left"])
            elif self.islr_head == "right":
                isolated_logits = pool_time(head_outputs["right"])
            elif self.islr_head == "body":
                isolated_logits = pool_time(head_outputs["body"])
            else:
                isolated_logits = pool_time(head_outputs["gloss_logits"])

            outputs = {
                **head_outputs,
                "isolated_logits": isolated_logits,
                "total_loss": 0,
            }

            if "gloss_label_isolated" in src_input:
                labels_isolated = src_input["gloss_label_isolated"].long()
            elif "gloss_labels" in src_input and src_input["gloss_labels"].ndim == 1:
                labels_isolated = src_input["gloss_labels"].long()
            else:
                labels_isolated = src_input["gloss_labels"][:, 0].long()

            isolated_loss = self.compute_isolated_loss(
                logits=isolated_logits, labels=labels_isolated
            )
            outputs["isolated_loss"] = isolated_loss
            outputs["total_loss"] += isolated_loss

            if self.self_distillation and self.distillation_on_islr:
                teacher_logits = isolated_logits.detach()
                for student, weight in self.cfg["distillation_weight"].items():
                    student_seq_logits = head_outputs[
                        "left"
                        if student == "left"
                        else (
                            "right"
                            if student == "right"
                            else ("body" if student == "body" else "gloss_logits")
                        )
                    ]
                    student_logits = pool_time(student_seq_logits)

                    if torch.isnan(student_logits).any():
                        raise ValueError(f"NaN in student_logits for {student}")
                    if torch.isnan(teacher_logits).any():
                        raise ValueError(f"NaN in teacher_logits for {student}")

                    distill_loss = weight * self.distillation_loss(
                        student_logits.unsqueeze(0),
                        teacher_logits.unsqueeze(0),
                        use_blank=False,
                    )
                    distill_loss = torch.clamp(distill_loss, min=-100, max=100)
                    outputs[f"{student}_distill_loss"] = distill_loss

                    if torch.isnan(outputs[f"{student}_distill_loss"]) or torch.isinf(
                        outputs[f"{student}_distill_loss"]
                    ):
                        print(
                            f"Distillation loss for {student}: {outputs[f'{student}_distill_loss']}"
                        )
                        raise ValueError(f"NaN or inf in {student}_distill_loss")

                    outputs["total_loss"] += outputs[f"{student}_distill_loss"]

            if return_attention_maps:
                attention_data["fusion_attention_data"] = fusion_attn_data
                outputs["attention_data"] = attention_data

            return outputs
        else:  # Continuous
            head_outputs = self.recognition_head(
                left_embed, right_embed, fuse_embed, body_embed
            )

            for k, v in head_outputs.items():
                if torch.isnan(v).any():
                    raise ValueError(f"NaN in {k}")
                if torch.isinf(v).any():
                    raise ValueError(f"inf in {k}")
            outputs = {
                **head_outputs,
                "input_lengths": src_input["valid_len_in"],
                "total_loss": 0,
            }

            if return_attention_maps:
                attention_data["fusion_attention_data"] = fusion_attn_data
                outputs["attention_data"] = attention_data

            outputs["fuse_loss"] = self.compute_loss(
                labels=src_input["gloss_labels"],
                tgt_lengths=src_input["gloss_lengths"],
                logits=outputs["gloss_logits"],
                input_lengths=outputs["input_lengths"],
            )

            outputs["total_loss"] += outputs["fuse_loss"]

            if self.self_distillation:
                for student, weight in self.cfg["distillation_weight"].items():
                    teacher_logits = outputs["gloss_logits"]
                    teacher_logits = teacher_logits.detach()
                    student_logits = outputs[f"{student}"]

                    if torch.isnan(student_logits).any():
                        raise ValueError(f"NaN in student_logits for {student}")

                    if torch.isnan(teacher_logits).any():
                        raise ValueError(f"NaN in teacher_logits for {student}")

                    distill_loss = weight * self.distillation_loss(
                        student_logits, teacher_logits, use_blank=False
                    )

                    distill_loss = torch.clamp(distill_loss, min=-100, max=100)
                    outputs[f"{student}_distill_loss"] = distill_loss

                    if torch.isnan(outputs[f"{student}_distill_loss"]) or torch.isinf(
                        outputs[f"{student}_distill_loss"]
                    ):
                        print(
                            f"Distillation loss for {student}: {outputs[f'{student}_distill_loss']}"
                        )
                        raise ValueError(f"NaN or inf in {student}_distill_loss")

                    outputs["total_loss"] += outputs[f"{student}_distill_loss"]

            return outputs

    def compute_loss(self, labels, tgt_lengths, logits, input_lengths):
        logits = logits.permute(1, 0, 2)

        log_probs = F.log_softmax(logits, dim=-1)

        log_probs = torch.clamp(log_probs, min=-100, max=0)

        if torch.isnan(log_probs).any():
            raise ValueError("NaN in log_probs")
        if torch.isinf(log_probs).any():
            raise ValueError("inf in log_probs")

        input_lengths = torch.clamp(input_lengths, min=1)
        tgt_lengths = torch.clamp(tgt_lengths, min=1)

        input_lengths = torch.maximum(input_lengths, tgt_lengths)

        loss = self.loss_fn(
            log_probs,
            labels.cpu().int(),
            input_lengths.cpu().int(),
            tgt_lengths.cpu().int(),
        )

        valid_loss_mask = torch.isfinite(loss)
        if valid_loss_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = loss[valid_loss_mask].mean()

        return loss

    def compute_isolated_loss(self, logits, labels):
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or inf in isolated logits")
        loss = self.isolated_ce(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN or inf in isolated loss")
        return loss

    def interface(self, keypoints, mask):
        body_embed, _ = self.body_encoder(
            keypoints[:, :, self.cfg["body_idx"], :],
            mask,
            return_attn_map=False,
        )
        left_embed, _ = self.left_encoder(
            keypoints[:, :, self.cfg["left_idx"], :],
            mask,
            return_attn_map=False,
        )
        right_embed, _ = self.right_encoder(
            keypoints[:, :, self.cfg["right_idx"], :],
            mask,
            return_attn_map=False,
        )

        fuse_embed = self.coordinates_fusion(left_embed, right_embed, body_embed)

        head_outputs = self.recognition_head(
            left_embed, right_embed, fuse_embed, body_embed
        )

        if self.task == "isolated":

            def pool_time(x):
                if self.islr_pooling == "max":
                    return torch.amax(x, dim=1)
                return torch.mean(x, dim=1)

            if self.islr_head == "left":
                isolated_logits = pool_time(head_outputs["left"])
            elif self.islr_head == "right":
                isolated_logits = pool_time(head_outputs["right"])
            elif self.islr_head == "body":
                isolated_logits = pool_time(head_outputs["body"])
            else:
                isolated_logits = pool_time(head_outputs["gloss_logits"])

            return {**head_outputs, "isolated_logits": isolated_logits}

        return head_outputs
