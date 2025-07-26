import torch
import json
from collections import defaultdict
import datetime
import math
import sys
import os
from pathlib import Path
from metrics import wer_list

from logger import MetricLogger, SmoothedValue
from utils import ctc_decode


def train_one_epoch(
    args, model, data_loader, optimizer, epoch, print_freq=1, log_dir="log/train"
):
    model.train()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metric_logger = MetricLogger(
        delimiter="  ",
        log_dir=log_dir,
        file_name=f"epoch_{epoch}_({timestamp}).log",
    )
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Training epoch: [{epoch}/{args.epochs}]"
    for step, (src_input) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer.zero_grad()
        output = model(src_input)
        loss = output["total_loss"]
        with torch.autograd.set_detect_anomaly(True):
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                print("NaN loss")
        model.zero_grad()
        for k, v in output.items():
            if "loss" in k and "gloss" not in k:
                # print(k, v)
                metric_logger.update(**{k: v})
        # metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged results:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def generate_attention_maps(
    args,
    dataloader,
    model,
    epoch,
    output_dir="./attention_maps",
    max_samples=100,
    tokenizer=None,
):
    """
    Generate and save attention maps for samples from the dataloader
    Args:
        args: command line arguments
        dataloader: data loader for evaluation
        model: trained model
        epoch: current epoch
        output_dir: directory to save attention maps
        max_samples: maximum number of samples to process
        tokenizer: tokenizer for decoding
    """
    print(f"Generating attention maps...")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.eval()

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save attention data
    attention_data_path = output_path / f"attention_data_{timestamp}.pt"

    samples_processed = 0
    batch_idx = 0

    with torch.no_grad():
        for batch_idx, src_input in enumerate(dataloader):
            if samples_processed >= max_samples:
                break

            print(
                f"Processing batch {batch_idx + 1}, samples {samples_processed + 1}-{min(samples_processed + args.batch_size, max_samples)}"
            )

            # Get model output with attention maps
            output = model(src_input, return_attention_maps=True)

            if "attention_data" not in output:
                print("Warning: No attention data found in model output")
                continue

            # Save batch attention data
            batch_attention_data = output["attention_data"]
            batch_save_path = output_path / f"batch_{batch_idx:04d}_attention_data.pt"
            torch.save(batch_attention_data, batch_save_path)
            print(f"Saved batch attention data: {batch_save_path}")

            # Generate visualizations for this batch
            print(f"Generating visualizations for batch {batch_idx}...")
            try:
                from visualize_attention import process_batch_attention

                print("Successfully imported visualize_attention module")

                process_batch_attention(
                    batch_attention_data, output_dir, batch_start_idx=samples_processed
                )
                print(f"Completed visualizations for batch {batch_idx}")

            except ImportError as e:
                print(f"ImportError: Could not import visualize_attention: {e}")
                print(
                    "Please ensure visualize_attention.py is in the project root directory"
                )
                print("And install required packages: pip install matplotlib seaborn")

            except Exception as e:
                print(f"Error during visualization: {e}")
                print("Full traceback:")
                import traceback

                traceback.print_exc()
                print("Continuing with next batch...")

            # Update sample count
            batch_size = src_input["keypoints"].shape[0]
            samples_processed += batch_size

            # Save sample metadata
            sample_metadata = {
                "names": src_input.get(
                    "name", [f"sample_{i}" for i in range(batch_size)]
                ),
                "gloss_references": src_input.get("gloss_input", []),
                "batch_idx": batch_idx,
                "samples_processed": samples_processed,
            }

            metadata_path = output_path / f"batch_{batch_idx:04d}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(sample_metadata, f, indent=2)

    print(f"Attention map generation completed!")
    print(f"Total samples processed: {samples_processed}")
    print(f"Total batches processed: {batch_idx + 1}")
    print(f"Results saved to: {output_dir}")

    return {
        "samples_processed": samples_processed,
        "batches_processed": batch_idx + 1,
        "output_dir": str(output_dir),
    }


def evaluate_fn(
    args,
    dataloader,
    model,
    epoch,
    beam_size=1,
    print_freq=1,
    results_path=None,
    tokenizer=None,
    log_dir="log/test",
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.eval()
    metric_logger = MetricLogger(
        log_dir=log_dir, file_name=f"epoch_{epoch}_({timestamp}).log"
    )
    header = f"Test epoch: [{epoch}/{args.epochs}]"
    print_freq = 10
    results = defaultdict(dict)

    with torch.no_grad():
        for _, (src_input) in enumerate(
            metric_logger.log_every(dataloader, print_freq, header)
        ):
            output = model(src_input)

            for k, gls_logits in output.items():
                if "_loss" in k and "gloss" not in k:
                    metric_logger.update(**{k: gls_logits})
                    continue
                if "gloss_logits" not in k:
                    continue
                logits_name = k.replace("gloss_logits", "")

                ctc_decode_output = ctc_decode(
                    gloss_logits=gls_logits,
                    beam_size=beam_size,
                    input_lengths=output["input_lengths"],
                )
                batch_pred_gls = tokenizer.batch_decode(ctc_decode_output)
                lower_case = tokenizer.lower_case
                for name, gls_hyp, gls_ref in zip(
                    src_input["name"], batch_pred_gls, src_input["gloss_input"]
                ):
                    results[name][f"{logits_name}_gls_hyp"] = (
                        gls_hyp.upper() if lower_case else gls_hyp
                    )
                    results[name]["gls_ref"] = (
                        gls_ref.upper() if lower_case else gls_ref
                    )

            metric_logger.update(loss=output["total_loss"].item())

        evaluation_results = {}
        evaluation_results["wer"] = 200

        for hyp_name in results[name].keys():
            if "gls_hyp" not in hyp_name:
                continue
            k = hyp_name.replace("gls_hyp", "")
            gls_ref = [results[n]["gls_ref"] for n in results]
            gls_hyp = [results[n][hyp_name] for n in results]
            wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
            # evaluation_results[k + "wer_list"] = wer_results
            evaluation_results[k + "wer"] = wer_results["wer"]
            metric_logger.update(**{k + "wer": wer_results["wer"]})
            evaluation_results["wer"] = min(
                wer_results["wer"], evaluation_results["wer"]
            )
        metric_logger.update(wer=evaluation_results["wer"])
        if results_path is not None:
            with open(results_path, "w") as f:
                json.dump(results, f)
    for k, v in evaluation_results.items():
        print(f"{k}: {v:.3f}")
    print("* Averaged results:", metric_logger)
    print("* DEV loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
