import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import yaml
import argparse


def save_attention_heatmap(
    attention_weights, save_path, title="Attention Map", sample_idx=0
):
    """
    Save attention weights as heatmap
    Args:
        attention_weights: torch.Tensor of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len)
        save_path: path to save the image
        title: title for the file naming (not displayed on plot)
        sample_idx: which sample in batch to visualize (default: 0)
    """
    if attention_weights is None:
        return

    # Convert to numpy and select first sample if batch
    if len(attention_weights.shape) == 3:
        attn_map = attention_weights[sample_idx].detach().cpu().numpy()
    else:
        attn_map = attention_weights.detach().cpu().numpy()

    # Create clean heatmap without title, labels, or ticks

    attn_map = attn_map[:32, :32]
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        attn_map,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )

    # Remove all axes, titles, and labels for clean image
    plt.axis("off")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save plot with no padding and high DPI
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,  # Remove padding
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    print(f"Saved attention map: {save_path}")


def save_attention_maps_for_sample(attention_data, sample_idx, sample_name, output_dir):
    """
    Save all attention maps for a single sample
    Args:
        attention_data: dict containing attention weights from different modules
        sample_idx: index of sample in batch
        sample_name: name/identifier for the sample
        output_dir: base output directory
    """
    sample_dir = Path(output_dir) / f"sample_{sample_name}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save different types of attention maps
    for body_part in ["body", "left", "right"]:
        if f"{body_part}_attention_data" in attention_data:
            part_data = attention_data[f"{body_part}_attention_data"]

            # X attention maps (self attention)
            if "self_attn_maps" in part_data and part_data["self_attn_maps"]:
                for layer_idx, self_attn in enumerate(
                    part_data["self_attn_maps"][0][0]
                ):
                    print(layer_idx, self_attn.shape)
                    file_name = f"{body_part}_x_attention_layer_{layer_idx}_sample_{sample_name}.png"
                    save_path = sample_dir / file_name
                    title = f"{body_part.capitalize()} X Attention Layer {layer_idx} Sample {sample_name}"
                    save_attention_heatmap(
                        self_attn, save_path, title=title, sample_idx=sample_idx
                    )

            # Y attention maps (causal attention)
            if "causal_attn_maps" in part_data and part_data["causal_attn_maps"]:
                for layer_idx, causal_attn in enumerate(
                    part_data["causal_attn_maps"][0][0]
                ):
                    file_name = f"{body_part}_y_attention_layer_{layer_idx}_sample_{sample_name}.png"
                    save_path = sample_dir / file_name
                    title = f"{body_part.capitalize()} Y Attention Layer {layer_idx} Sample {sample_name}"
                    save_attention_heatmap(
                        causal_attn, save_path, title=title, sample_idx=sample_idx
                    )

            # Cross attention maps
            if "cross_attn_maps" in part_data and part_data["cross_attn_maps"]:
                for layer_idx, cross_attn in enumerate(
                    part_data["cross_attn_maps"][0][0]
                ):
                    file_name = f"{body_part}_cross_attention_layer_{layer_idx}_sample_{sample_name}.png"
                    save_path = sample_dir / file_name
                    title = f"{body_part.capitalize()} Cross Attention Layer {layer_idx} Sample {sample_name}"
                    save_attention_heatmap(
                        cross_attn, save_path, title=title, sample_idx=sample_idx
                    )


def save_summary_attention_maps(attention_data, sample_idx, sample_name, output_dir):
    """
    Save averaged attention maps across layers for summary view
    """
    sample_dir = Path(output_dir) / f"sample_{sample_name}"
    summary_dir = sample_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for body_part in ["body", "left", "right"]:
        if f"{body_part}_attention_data" in attention_data:
            part_data = attention_data[f"{body_part}_attention_data"]

            # Average self attention across layers
            if "self_attn_maps" in part_data and part_data["self_attn_maps"]:
                avg_self_attn = torch.stack(part_data["self_attn_maps"]).mean(dim=0)
                file_name = f"{body_part}_x_attention_avg_sample_{sample_name}.png"
                save_path = summary_dir / file_name
                title = (
                    f"{body_part.capitalize()} X Attention Average Sample {sample_name}"
                )
                save_attention_heatmap(
                    avg_self_attn, save_path, title=title, sample_idx=sample_idx
                )

            # Average causal attention across layers
            if "causal_attn_maps" in part_data and part_data["causal_attn_maps"]:
                avg_causal_attn = torch.stack(part_data["causal_attn_maps"]).mean(dim=0)
                file_name = f"{body_part}_y_attention_avg_sample_{sample_name}.png"
                save_path = summary_dir / file_name
                title = (
                    f"{body_part.capitalize()} Y Attention Average Sample {sample_name}"
                )
                save_attention_heatmap(
                    avg_causal_attn, save_path, title=title, sample_idx=sample_idx
                )

            # Average cross attention across layers
            if "cross_attn_maps" in part_data and part_data["cross_attn_maps"]:
                avg_cross_attn = torch.stack(part_data["cross_attn_maps"]).mean(dim=0)
                file_name = f"{body_part}_cross_attention_avg_sample_{sample_name}.png"
                save_path = summary_dir / file_name
                title = f"{body_part.capitalize()} Cross Attention Average Sample {sample_name}"
                save_attention_heatmap(
                    avg_cross_attn, save_path, title=title, sample_idx=sample_idx
                )


def process_batch_attention(attention_data, output_dir, batch_start_idx=0):
    """
    Process attention maps for entire batch
    Args:
        attention_data: dict containing attention data for the batch
        output_dir: output directory
        batch_start_idx: starting index for sample naming
    """
    # Determine batch size from the first available attention tensor
    batch_size = None
    for body_part in ["body", "left", "right"]:
        if f"{body_part}_attention_data" in attention_data:
            part_data = attention_data[f"{body_part}_attention_data"]
            for attn_type in [
                "self_attn_maps",
                "causal_attn_maps",
                "cross_attn_maps",
            ]:
                if attn_type in part_data and part_data[attn_type]:
                    batch_size = part_data[attn_type][0].shape[0]
                    break
            if batch_size is not None:
                break

    if batch_size is None:
        print("Warning: Could not determine batch size from attention data")
        return

    # Process each sample in the batch
    for sample_idx in range(batch_size):
        sample_name = f"{batch_start_idx + sample_idx:06d}"

        print(f"Processing sample {sample_name} (batch index {sample_idx})")

        # Save detailed attention maps
        save_attention_maps_for_sample(
            attention_data, sample_idx, sample_name, output_dir
        )

        # Save summary attention maps
        save_summary_attention_maps(attention_data, sample_idx, sample_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention maps")
    parser.add_argument(
        "--attention_data",
        required=True,
        help="Path to saved attention data (.pt file)",
    )
    parser.add_argument(
        "--output_dir", default="./attention_visualizations", help="Output directory"
    )
    parser.add_argument(
        "--batch_start_idx",
        type=int,
        default=0,
        help="Starting index for sample naming",
    )

    args = parser.parse_args()

    # Load attention data
    print(f"Loading attention data from {args.attention_data}")
    attention_data = torch.load(args.attention_data, map_location="cpu")

    # Process and save visualizations
    print(f"Saving visualizations to {args.output_dir}")
    process_batch_attention(attention_data, args.output_dir, args.batch_start_idx)

    print("Attention visualization completed!")
