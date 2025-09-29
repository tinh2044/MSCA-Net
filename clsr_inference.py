import argparse
import json
import os

import torch

from Tokenizer import GlossTokenizer
from model import MSCA_Net
from utils.inference import (
    load_config,
    load_checkpoint,
    build_class_mappings,
    preprocess_sample,
)
from utils.keypoint import extract_from_video
from utils import ctc_decode


def run_inference(
    video_path,
    cfg_path,
    checkpoint_path,
    device="cpu",
    beam_size=5,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    tokenizer = GlossTokenizer(cfg["gloss_tokenizer"])

    model = MSCA_Net(cfg=model_cfg, gloss_tokenizer=tokenizer, device=device)
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
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MSCA-Net CSLR inference")
    parser.add_argument(
        "--cfg",
        default="./configs/phoenix-2014t.yaml",
        type=str,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        default="./outputs_Phoenix-2014-T/best.pth",
        type=str,
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--video", default="./sample.mp4", type=str, help="Path to input video file"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Compute device"
    )
    parser.add_argument("--beam-size", type=int, default=5, help="CTC beam size")

    args = parser.parse_args()
    result = run_inference(
        video_path=args.video,
        cfg_path=args.cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        beam_size=args.beam_size,
    )
    print()
    print(
        f"Prediction gloss sentence: {result['prediction']} with time steps {result['time_steps']}"
    )
