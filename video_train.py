import os
import time
import argparse
import datetime
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ISLRVideoDataset, video_collate_fn
from optimizer import build_optimizer, build_scheduler
import utils

from video_model import build_i3d, build_slowfast, TimeSFormerWrapper

from utils.video_opt import train_one_epoch, evaluate_fn


def get_args_parser():
    parser = argparse.ArgumentParser("ISLR Video Training", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    return parser


def build_model_from_cfg(cfg):
    mcfg = cfg["model"]
    mtype = mcfg.get("type", "i3d").lower()
    num_classes = mcfg["num_classes"]

    if mtype == "i3d":
        return build_i3d(
            num_classes=num_classes, pretrained=mcfg.get("pretrained", True)
        )
    elif mtype == "slowfast":
        return build_slowfast(
            num_classes=num_classes,
            pretrained=mcfg.get("pretrained", True),
            alpha=mcfg.get("alpha", 4),
        )
    elif mtype == "timesformer":
        return TimeSFormerWrapper(
            num_classes=num_classes,
            pretrained=mcfg.get("pretrained", True),
            attention_type=mcfg.get("attention_type", "divided_space_time"),
            img_size=mcfg.get("img_size", 224),
            num_frames=mcfg.get("num_frames", 16),
        )
    else:
        raise ValueError(f"Unknown model.type: {mtype}")


def main(args, cfg):
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    args.device = device

    model_dir = cfg["training"]["model_dir"]
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"{model_dir}/train_log_{timestamp}.log"

    data_cfg = cfg["data"]
    mcfg = cfg["model"]
    num_frames = mcfg.get(
        "num_frames", 32 if mcfg.get("type", "i3d").lower() != "timesformer" else 16
    )
    img_size = mcfg.get("img_size", 224)

    train_set = ISLRVideoDataset(
        root=data_cfg["root"],
        split="train",
        num_frames=num_frames,
        size=img_size,
        shuffle=True,
    )
    test_set = ISLRVideoDataset(
        root=data_cfg["root"],
        split="test",
        num_frames=num_frames,
        size=img_size,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=video_collate_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=video_collate_fn,
    )

    model = build_model_from_cfg(cfg)
    model = model.to(device)
    print(f"Number of parameters: {utils.count_model_parameters(model)}")

    optim_cfg = cfg["training"]["optimization"]
    optimizer = build_optimizer(config=optim_cfg, model=model)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]
    cfg["training"]["optimization"]["total_epochs"] = args.epochs

    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"], optimizer=optimizer, last_epoch=-1
    )
    print(f"Scheduler: {scheduler_type}")

    if args.resume:
        print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if utils.check_state_dict(model, checkpoint["model"]):
            ret = model.load_state_dict(checkpoint["model"], strict=False)
        else:
            raise ValueError("Model and state dict are different")
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))
        if not args.eval and "optimizer" in checkpoint and "scheduler" in checkpoint:
            print("Loading optimizer and scheduler")
            optimizer.load_state_dict(checkpoint["optimizer"])
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"New learning rate : {scheduler.get_last_lr()[0]}")

    output_dir = Path(model_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.eval:
        results = evaluate_fn(
            args,
            test_loader,
            model,
            epoch=0,
            print_freq=args.print_freq,
            log_file=f"{model_dir}/eval.log",
        )
        print(f"ACC {results.get('acc', 0.0):.4f}")
        return

    print(f"Training on {device}")
    print(
        f"Start training for {args.epochs} epochs and start epoch: {args.start_epoch}"
    )
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        _ = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            epoch,
            print_freq=args.print_freq,
            log_file=log_file,
        )
        scheduler.step()

        # save last
        last_ckpt = output_dir / f"checkpoint_{epoch}.pth"
        utils.save_on_master(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            last_ckpt,
        )
        prev = output_dir / f"checkpoint_{epoch - 1}.pth"
        if os.path.exists(prev):
            os.remove(prev)

        results = evaluate_fn(
            args,
            test_loader,
            model,
            epoch,
            print_freq=args.print_freq,
            log_file=f"{model_dir}/eval.log",
        )
        acc = results.get("acc", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_path = output_dir / "best_checkpoint.pth"
            utils.save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                best_path,
            )
        print(f"ACC {acc:.4f} | BEST_ACC {best_acc:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("ISLR Video Training", parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.cfg, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    Path(config["training"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    main(args, config)
