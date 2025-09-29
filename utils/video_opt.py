import torch
from utils.logger import MetricLogger, SmoothedValue
from utils import update_islr_metrics, compute_islr_metrics


def train_one_epoch(args, model, data_loader, optimizer, epoch, print_freq, log_file):
    model.train()
    metric_logger = MetricLogger(delimiter=", ", log_file=log_file)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=10, fmt="{value:.4f}"))

    header = f"Training epoch: [{epoch}/{args.epochs}]"

    num_classes = getattr(model, "num_classes", 0)
    conf_mat = (
        [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        if num_classes and num_classes > 0
        else None
    )

    for _, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if "clip" in batch:
            inputs = batch["clip"].to(args.device)
        else:
            inputs = batch.get("inputs").to(args.device)
        labels = batch["labels"].to(args.device)

        optimizer.zero_grad()
        output = model(inputs, labels)
        loss = output["total_loss"]

        if torch.isfinite(loss).all().item():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            raise ValueError("Non-finite loss detected")

        if conf_mat is not None and "isolated_logits" in output:
            preds = output["isolated_logits"].argmax(dim=-1)
            update_islr_metrics(conf_mat, preds, labels, num_classes)

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if conf_mat is not None:
        acc, precision, recall = compute_islr_metrics(conf_mat, num_classes)
        results.update({"acc": acc, "precision": precision, "recall": recall})
    print("Averaged results:", metric_logger)
    return results


def evaluate_fn(args, data_loader, model, epoch, print_freq, log_file):
    model.eval()
    metric_logger = MetricLogger(delimiter=", ", log_file=log_file)
    header = f"Test epoch: [{epoch}/{args.epochs}]"

    num_classes = getattr(model, "num_classes", 0)
    conf_mat = (
        [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        if num_classes and num_classes > 0
        else None
    )

    with torch.no_grad():
        for _, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            if "clip" in batch:
                inputs = batch["clip"].to(args.device)
            else:
                inputs = batch.get("inputs").to(args.device)
            labels = batch["labels"].to(args.device)

            output = model(inputs, labels)
            if "total_loss" in output:
                metric_logger.update(loss=output["total_loss"].item())

            if conf_mat is not None and "isolated_logits" in output:
                preds = output["isolated_logits"].argmax(dim=-1)
                update_islr_metrics(conf_mat, preds, labels, num_classes)

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
