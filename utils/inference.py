import os
import numpy as np
import torch
import yaml


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def select_frames(keypoints, max_len):
    n_frames = keypoints.shape[0]
    if n_frames <= max_len:
        return keypoints
    frames_idx = np.arange(n_frames)
    f_s = (n_frames - max_len) // 2
    f_e = n_frames - max_len - f_s
    frames_idx = frames_idx[f_s:-f_e]
    assert len(frames_idx) <= max_len
    return keypoints[frames_idx]


def normalize_part(part_xy):
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


def normalize_per_parts(keypoints_xy, joint_parts):
    for i in range(keypoints_xy.shape[0]):
        for part in joint_parts:
            keypoints_xy[i, part, :2] = normalize_part(keypoints_xy[i, part, :2])
    return keypoints_xy


def preprocess_sample(raw_keypoints, data_cfg):
    assert raw_keypoints.ndim == 3 and raw_keypoints.shape[-1] >= 2
    keypoints_xy = raw_keypoints[:, :, :2].astype(np.float32)

    keypoints_xy = select_frames(keypoints_xy, data_cfg["max_len"])

    if data_cfg.get("normalize", True):
        keypoints_xy = normalize_per_parts(keypoints_xy, data_cfg["joint_parts"])

    T = keypoints_xy.shape[0]
    mask = torch.ones(1, T, dtype=torch.long)

    keypoints_tensor = torch.from_numpy(keypoints_xy).unsqueeze(0).float()
    return keypoints_tensor, mask


def load_checkpoint(model, ckpt_path):
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
        raise ValueError("Unsupported checkpoint format")

    ret = model.load_state_dict(state_dict, strict=False)
    missing = "\n".join(ret.missing_keys)
    unexpected = "\n".join(ret.unexpected_keys)
    if missing:
        print("Missing keys:\n" + missing)
    if unexpected:
        print("Unexpected keys:\n" + unexpected)
