#!/usr/bin/env python3
"""
Script to visualize attention maps from MSCA-Net model.
This script extracts and saves attention maps for each sample in the dataset.
"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import argparse
import json
import numpy as np
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil

from Tokenizer import GlossTokenizer
from dataset import SLR_Dataset
from model import MSCA_Net
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("MSCA-Net Attention Visualization", add_help=False)
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for visualization")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./attention_maps", help="Output directory for attention maps")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Dataset split to visualize")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", default=42, type=int)
    return parser


def save_attention_map(attention_map, save_path, title="Attention Map"):
    """
    Save attention map as PNG image without labels, legend, and title.
    Different colors for different attention types.

    Args:
        attention_map: Tensor of shape (T, T) - attention weights
        save_path: Path to save the image
        title: Title for the plot (not used, kept for compatibility)
    """
    # Convert to numpy and ensure it's on CPU
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()

    # DEBUG: Print detailed statistics
    filename = os.path.basename(save_path)


    attention_map = attention_map

    # Normalize attention map to [0, 1] range for better visualization
    original_min, original_max = attention_map.min(), attention_map.max()
  
    # Choose colormap based on attention type from filename
    filename = os.path.basename(save_path)
    if "self_attn_maps" in filename:
        colormap = 'Greens'  # Xanh lá cho self attention
    elif "causal_attn_maps" in filename:
        colormap = 'Oranges'  # Cam cho causal attention
    elif "cross_attn_maps" in filename:
        colormap = 'RdPu'  # Hồng cho cross attention (Red-Purple)
    else:
        colormap = 'plasma'  # Default colormap

    # Create figure
    plt.figure(figsize=(10, 10))

    # Create heatmap using seaborn with attention-specific colormap
    sns.heatmap(
        attention_map,
        cmap=colormap,  # Use colormap based on attention type
        cbar=False,  # No colorbar
        square=True,  # Square cells
        xticklabels=False,  # No x labels
        yticklabels=False,  # No y labels
        annot=False,  # No annotations
        fmt='.10f',
        vmin=original_min,  # Set minimum value for consistent color scaling
        vmax=original_max  # Set maximum value for consistent color scaling
    )

    # Remove axes and labels
    plt.axis('off')

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_attention_data(attention_data, sample_names, output_dir, data_name):
    """
    Process and save attention maps for a batch of samples.

    Args:
        attention_data: Dictionary containing attention maps for all streams
        sample_names: List of sample names in the batch
        output_dir: Base output directory
        data_name: Name of the dataset
    """
    batch_size = len(sample_names)

    # Define the streams and attention types
    streams = ["body_attention_data", "left_attention_data", "right_attention_data"]
    attention_types = ["self_attn_maps", "causal_attn_maps", "cross_attn_maps"]
    for batch_idx in range(batch_size):
        sample_name = sample_names[batch_idx]
        sample_dir = os.path.join(output_dir, data_name, sample_name)

        print(f"\nProcessing sample: {sample_name}")

        for stream in streams:
            if stream not in attention_data or attention_data[stream] is None:
                print(f"  Warning: {stream} not found in attention data")
                continue

            stream_data = attention_data[stream]
            stream_name = stream.replace("_attention_data", "")  # body, left, right

            for attn_type in attention_types:
                if attn_type not in stream_data or stream_data[attn_type] is None:
                    print(f"  Warning: {attn_type} not found in {stream}")
                    continue

                # Get attention map for current sample
                attn_map = stream_data[attn_type][batch_idx]  # Shape: (T, T)

                # Create filename
                filename = f"{stream_name}_{attn_type}.png"
                save_path = os.path.join(sample_dir, filename)

                # Save attention map (no title needed)
                save_attention_map(attn_map, save_path)

                print(f"    Saved: {filename} (shape: {attn_map.shape})")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Load config
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    
    # Initialize tokenizer
    gloss_tokenizer = GlossTokenizer(config["gloss_tokenizer"])
    
    # Load dataset
    cfg_data = config["data"]
    dataset = SLR_Dataset(
        root=cfg_data["root"],
        gloss_tokenizer=gloss_tokenizer,
        cfg=cfg_data,
        split=args.split,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 for debugging
        collate_fn=dataset.data_collator,
        shuffle=False,
        pin_memory=True,
    )
    
    # Initialize model
    model = MSCA_Net(cfg=config["model"], gloss_tokenizer=gloss_tokenizer, device=device)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset name from config or use default
    data_name = cfg_data.get("dataset_name", f"dataset_{args.split}")
    if os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    print(f"Starting attention visualization for {data_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing {args.split} split")
    
    # Process samples
    processed_samples = 0
    with torch.no_grad():
        for batch_idx, src_input in enumerate(tqdm(dataloader, desc="Processing batches")):
            if args.max_samples and processed_samples >= args.max_samples:
                break
                
            # Move to device
            if device == "cuda":
                src_input = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in src_input.items()
                }
            
            # Forward pass with attention maps
            outputs = model(src_input, return_attention_maps=True)
            attention_data = outputs["attention_data"]
            sample_names = src_input["name"]
            
            # Process and save attention maps
            process_attention_data(attention_data, sample_names, args.output_dir, data_name)
            
            processed_samples += len(sample_names)
                
            # except Exception as e:
            #     print(f"Error processing batch {batch_idx}: {str(e)}")
            #     continue
    
    print(f"Completed! Processed {processed_samples} samples.")
    print(f"Attention maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
