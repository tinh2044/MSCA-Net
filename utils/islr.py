import torch
from utils.logger import MetricLogger, SmoothedValue
from utils import update_islr_metrics, compute_islr_metrics


def train_one_epoch(
    args, model, data_loader, optimizer, epoch, print_freq=1, log_file=""
):
    model.train()
    metric_logger = MetricLogger(
        delimiter="  ",
        log_file=log_file,
    )
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "isolated_loss", SmoothedValue(window_size=10, fmt="{value:.4f}")
    )
    header = f"Training epoch: [{epoch}/{args.epochs}]"

    num_classes = model.num_classes
    conf_mat = (
        [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        if num_classes > 0
        else None
    )

    for step, (src_input) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer.zero_grad()
        output = model(src_input)
        loss = output["total_loss"]
        with torch.autograd.set_detect_anomaly(True):
            if torch.isfinite(loss).all().item():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                raise ValueError("Non-finite loss detected")
        model.zero_grad()

        metric_logger.update(isolated_loss=loss)
        if conf_mat is not None and "isolated_logits" in output:
            preds = output["isolated_logits"].argmax(dim=-1)
            targets = src_input["gloss_label_isolated"].to(preds.device)
            update_islr_metrics(conf_mat, preds, targets, num_classes)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if conf_mat is not None:
        acc, precision, recall = compute_islr_metrics(conf_mat, num_classes)
        results.update({"acc": acc, "precision": precision, "recall": recall})
    print("Averaged results:", metric_logger)
    return results


def evaluate_fn(args, dataloader, model, epoch, print_freq=1, log_file="log/test"):
    model.eval()
    metric_logger = MetricLogger(
        log_file=log_file,
    )
    header = f"Test epoch: [{epoch}/{args.epochs}]"
    print_freq = 10

    num_classes = model.num_classes
    conf_mat = (
        [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        if num_classes > 0
        else None
    )

    with torch.no_grad():
        for _, (src_input) in enumerate(
            metric_logger.log_every(dataloader, print_freq, header)
        ):
            output = model(src_input)
            if "total_loss" in output:
                metric_logger.update(loss=output["total_loss"].item())

            if conf_mat is not None and "isolated_logits" in output:
                preds = output["isolated_logits"].argmax(dim=-1)
                targets = src_input["gloss_label_isolated"].to(preds.device)
                update_islr_metrics(conf_mat, preds, targets, num_classes)

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if conf_mat is not None:
        acc, precision, recall = compute_islr_metrics(conf_mat, num_classes)
        results.update({"acc": acc, "precision": precision, "recall": recall})

    print("Averaged results:", metric_logger)
    if conf_mat is not None:
        print(
            f"ACC {results['acc']:.4f} | PREC {results['precision']:.4f} | REC {results['recall']:.4f}"
        )
    return results
