import argparse
import json
import os

import torch

from utils.keypoint import extract_from_video
from utils.inference import (
    load_config,
    load_checkpoint,
    preprocess_sample,
)
from model import MSCA_Net

CLASS2ID = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}


def run_inference(
    video_path,
    cfg_path,
    checkpoint_path,
    device="cpu",
    classes_root=None,
    output_path=None,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"].copy()
    model = MSCA_Net(cfg=model_cfg, device=device, num_classes=model_cfg["num_classes"])
    model = model.to(device)
    model.eval()

    load_checkpoint(model, checkpoint_path)

    raw_keypoints = extract_from_video(video_path)
    if raw_keypoints is None or raw_keypoints.size == 0:
        raise ValueError("No keypoints extracted from the video.")

    keypoints_tensor, mask = preprocess_sample(raw_keypoints, data_cfg)
    keypoints_tensor = keypoints_tensor.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        head_outputs = model.interface(keypoints_tensor, mask)

    if "isolated_logits" not in head_outputs:
        raise RuntimeError(
            "Model did not return 'isolated_logits'. Ensure cfg.model.task='isolated'."
        )

    logits = head_outputs["isolated_logits"][0]  # [num_classes]
    pred_id = int(torch.argmax(logits).item())

    if classes_root is None:
        classes_root = data_cfg.get("root", ".")

    pred_label = CLASS2ID.get(pred_id, str(pred_id))

    result = {
        "video": video_path,
        "prediction_id": pred_id,
        "prediction_label": pred_label,
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MSCA-Net ISLR interface inference")
    parser.add_argument(
        "--cfg", default="./configs/vsl.yaml", type=str, help="Path to config YAML"
    )
    parser.add_argument(
        "--checkpoint",
        default="./outputs_VSL/best.pth",
        type=str,
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--video",
        default="./sample_islr.mp4",
        type=str,
        help="Path to input video file",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Compute device"
    )
    parser.add_argument(
        "--classes-root",
        type=str,
        default=None,
        help="Directory containing class subfolders (defaults to cfg.data.root/train or /test)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save JSON result"
    )
    args = parser.parse_args()
    result = run_inference(
        video_path=args.video,
        cfg_path=args.cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        classes_root=args.classes_root,
        output_path=args.output,
    )
    print(
        f"Predicted class: {result['prediction_label']} and id={result['prediction_id']}"
    )
