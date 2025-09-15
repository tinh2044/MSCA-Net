import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
import yaml

from Tokenizer import GlossTokenizer
from model import MSCA_Net
from scripts.keypoint_processer import extract_from_video
from utils import ctc_decode


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def select_frames_inference(keypoints: np.ndarray, max_len: int) -> np.ndarray:
    n_frames = keypoints.shape[0]
    if n_frames <= max_len:
        return keypoints
    frames_idx = np.arange(n_frames)
    f_s = (n_frames - max_len) // 2
    f_e = n_frames - max_len - f_s
    frames_idx = frames_idx[f_s:-f_e]
    assert len(frames_idx) <= max_len
    return keypoints[frames_idx]


def normalize_part_xy(part_xy: np.ndarray) -> np.ndarray:
    assert part_xy.shape[-1] == 2, "Keypoints must have x, y"
    x_coords = part_xy[:, 0]
    y_coords = part_xy[:, 1]
    min_x, min_y = float(np.min(x_coords)), float(np.min(y_coords))
    max_x, max_y = float(np.max(x_coords)), float(np.max(y_coords))
    w = max_x - min_x
    h = max_y - min_y

    if w > h:
        delta_x = 0.05 * w
        delta_y = delta_x + ((w - h) / 2)
    else:
        delta_y = 0.05 * h
        delta_x = delta_y + ((h - w) / 2)

    s_point = [max(0.0, min(min_x - delta_x, 1.0)), max(0.0, min(min_y - delta_y, 1.0))]
    e_point = [max(0.0, min(max_x + delta_x, 1.0)), max(0.0, min(max_y + delta_y, 1.0))]

    if (e_point[0] - s_point[0]) != 0.0:
        part_xy[:, 0] = (part_xy[:, 0] - s_point[0]) / (e_point[0] - s_point[0])
    if (e_point[1] - s_point[1]) != 0.0:
        part_xy[:, 1] = (part_xy[:, 1] - s_point[1]) / (e_point[1] - s_point[1])

    return part_xy


def normalize_keypoints_per_parts(
    keypoints_xy: np.ndarray, joint_parts: List[List[int]]
) -> np.ndarray:
    for i in range(keypoints_xy.shape[0]):
        for part in joint_parts:
            keypoints_xy[i, part, :2] = normalize_part_xy(keypoints_xy[i, part, :2])
    return keypoints_xy


def prepare_single_sample(
    raw_keypoints: np.ndarray,
    data_cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    assert raw_keypoints.ndim == 3 and raw_keypoints.shape[-1] >= 2
    keypoints_xy = raw_keypoints[:, :, :2].astype(np.float32)

    keypoints_xy = select_frames_inference(keypoints_xy, data_cfg["max_len"])

    if data_cfg.get("normalize", True):
        keypoints_xy = normalize_keypoints_per_parts(
            keypoints_xy, data_cfg["joint_parts"]
        )

    T = keypoints_xy.shape[0]
    mask = torch.ones(1, T, dtype=torch.long)

    keypoints_tensor = torch.from_numpy(keypoints_xy).unsqueeze(0).float()
    return keypoints_tensor, mask, T


def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str) -> None:
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = None
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            sample_key = next(iter(checkpoint.keys())) if len(checkpoint) > 0 else None
            if sample_key is not None and isinstance(
                checkpoint[sample_key], torch.Tensor
            ):
                state_dict = checkpoint
    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format. Expecting dict with 'model' or a state_dict."
        )
    ret = model.load_state_dict(state_dict, strict=False)
    missing = "\n".join(ret.missing_keys)
    unexpected = "\n".join(ret.unexpected_keys)
    if missing:
        print("Missing keys:\n" + missing)
    if unexpected:
        print("Unexpected keys:\n" + unexpected)


def run_inference(
    video_path: str,
    cfg_path: str,
    checkpoint_path: str,
    device: str = "cpu",
    beam_size: int = None,
    output_path: str = None,
) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    if beam_size is None:
        beam_size = int(
            cfg.get("testing", {}).get("recognition", {}).get("beam_size", 5)
        )

    tokenizer = GlossTokenizer(cfg["gloss_tokenizer"])

    selected_device = device
    if selected_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        selected_device = "cpu"
    model = MSCA_Net(cfg=model_cfg, gloss_tokenizer=tokenizer, device=selected_device)
    model = model.to(selected_device)
    model.eval()

    load_checkpoint_flex(model, checkpoint_path)

    raw_keypoints = extract_from_video(video_path)
    if raw_keypoints is None or raw_keypoints.size == 0:
        raise ValueError("No keypoints extracted from the video.")

    keypoints_tensor, mask, T = prepare_single_sample(raw_keypoints, data_cfg)
    keypoints_tensor = keypoints_tensor.to(selected_device)
    mask = mask.to(selected_device)

    with torch.no_grad():
        head_outputs = model.interface(keypoints_tensor, mask)

    gloss_logits = head_outputs["gloss_logits"]  # [B, T', V]
    time_steps = int(gloss_logits.shape[1])
    input_lengths = torch.tensor([time_steps], dtype=torch.long)

    decoded_ids = ctc_decode(
        gloss_logits=gloss_logits, beam_size=beam_size, input_lengths=input_lengths
    )
    predictions = tokenizer.batch_decode(decoded_ids)

    result = {
        "video": video_path,
        "prediction": predictions[0] if len(predictions) > 0 else "",
        "beam_size": beam_size,
        "time_steps": time_steps,
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MSCA-Net interface inference")
    parser.add_argument("--cfg", required=True, type=str, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--video", required=True, type=str, help="Path to input video file"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Compute device"
    )
    parser.add_argument(
        "--beam-size", type=int, default=None, help="CTC beam size (override config)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save JSON result"
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    result = run_inference(
        video_path=args.video,
        cfg_path=args.cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        beam_size=args.beam_size,
        output_path=args.output,
    )
    print(
        f"Prediction gloss sentence: {result['prediction']} with time steps {result['time_steps']}"
    )
