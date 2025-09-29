import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import argparse
import datetime
import numpy as np
import yaml
import random
from pathlib import Path

from loguru import logger

from optimizer import build_optimizer, build_scheduler
from dataset import ISLR_Dataset
from model import MSCA_Net

from utils.islr import train_one_epoch, evaluate_fn
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("MSCA-Net ISLR scripts", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume training from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    return parser


def main(args, cfg):
    model_dir = cfg["training"]["model_dir"]
    log_dir = f"{model_dir}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{log_dir}/islr_log_{timestamp}.log"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    cfg_data = cfg["data"]

    train_data = ISLR_Dataset(
        root=cfg_data["root"],
        cfg=cfg_data,
        split="train",
    )
    test_data = ISLR_Dataset(
        root=cfg_data["root"],
        cfg=cfg_data,
        split="test",
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.data_collator,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.data_collator,
        pin_memory=True,
    )

    model_cfg = cfg["model"].copy()
    model_cfg["task"] = model_cfg.get("task", "continuous")
    if model_cfg["task"] != "isolated":
        model_cfg["task"] = "isolated"

    model = MSCA_Net(cfg=model_cfg, device=device, num_classes=model_cfg["num_classes"])
    model = model.to(device)
    n_parameters = utils.count_model_parameters(model)
    print(f"Number of parameters: {n_parameters}")

    optimizer = build_optimizer(config=cfg["training"]["optimization"], model=model)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]
    cfg["training"]["optimization"]["total_epochs"] = args.epochs

    scheduler_last_epoch = -1
    if args.resume:
        print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if utils.check_state_dict(model, checkpoint["model"]):
            ret = model.load_state_dict(checkpoint["model"], strict=False)
        else:
            raise ValueError("Model and state dict are different")

        if "epoch" in checkpoint:
            scheduler_last_epoch = checkpoint["epoch"]
        args.start_epoch = checkpoint["epoch"] + 1
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    if args.finetune:
        print(f"Finetuning from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        ret = model.load_state_dict(checkpoint["model"], strict=False)
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"],
        optimizer=optimizer,
        last_epoch=scheduler_last_epoch,
    )
    logger.info(f"Scheduler: {scheduler_type}")

    if args.resume:
        if not args.eval and "optimizer" in checkpoint and "scheduler" in checkpoint:
            print("Loading optimizer and scheduler")
            optimizer.load_state_dict(checkpoint["optimizer"])
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"New learning rate : {scheduler.get_last_lr()[0]}")

    output_dir = Path(cfg["training"]["model_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.eval:
        results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch=0,
            print_freq=args.print_freq,
            log_file=f"{model_dir}/islr_eval.log",
        )
        print(
            f"ACC {results['acc']:.4f} | PREC {results['precision']:.4f} | REC {results['recall']:.4f}"
        )
        return

    print(f"Training on {device}")
    print(
        f"Start training for {args.epochs} epochs and start epoch: {args.start_epoch}"
    )
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_results = train_one_epoch(
            args,
            model,
            train_dataloader,
            optimizer,
            epoch,
            print_freq=args.print_freq,
            log_file=log_file,
        )
        scheduler.step()
        checkpoint_paths = [output_dir / f"islr_checkpoint_{epoch}.pth"]
        prev_chkpt = output_dir / f"islr_checkpoint_{epoch - 1}.pth"
        if os.path.exists(prev_chkpt):
            os.remove(prev_chkpt)
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch,
            print_freq=args.print_freq,
            log_file=f"{model_dir}/islr_eval.log",
        )

        if results.get("acc", 0.0) > best_acc:
            best_acc = results["acc"]
            checkpoint_paths = [output_dir / "best_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )

        print(
            f"ACC {results['acc']:.4f} | PREC {results['precision']:.4f} | REC {results['recall']:.4f} | BEST_ACC {best_acc:.4f}"
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        "MSCA-Net ISLR scripts", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    with open(args.cfg, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    Path(config["training"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    main(args, config)
