#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from celeba_nn_stage.checkpointing import (
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    find_best_resume,
    restore_rng_state,
)
from celeba_nn_stage.data import DEFAULT_BASE_REPO, DEFAULT_DATA_DIR, load_full_s1
from celeba_nn_stage.ema import ModelEMA
from celeba_nn_stage.model import ModelConfig, VelocityUNet80
from celeba_nn_stage.time_axis import write_time_axis_reference

DEFAULT_RUN_DIR = Path("/root/shared-nvme/experiment-real-data/runs/celeba_NN_stage_S1_t_uniform")
DEFAULT_MILESTONES = (1_000, 5_000, 20_000, 50_000, 100_000, 200_000)
FORMAT = "celeba80_cfm_t_uniform_velocity_v2"


def parse_int_csv(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(x.strip()) for x in text.split(",") if x.strip())))
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("checkpoint steps must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CelebA80 full-S1 conditional flow matching with t~U(0,1).")
    p.add_argument("--base-repo", type=Path, default=DEFAULT_BASE_REPO)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--checkpoint-steps", type=parse_int_csv, default=DEFAULT_MILESTONES)
    p.add_argument("--autosave-every", type=int, default=1_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--time-emb-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if supported; off by default for robustness.")
    p.add_argument("--no-auto-resume", action="store_true", help="Start from scratch even if checkpoints exist.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_signature(args: argparse.Namespace, model_cfg: ModelConfig, dataset_info: dict) -> dict:
    # max_steps and checkpoint cadence are intentionally excluded so a run may
    # later be extended without invalidating an otherwise identical checkpoint.
    return {
        "format": FORMAT,
        "group": "S1",
        "dataset_count": dataset_info["count"],
        "data_dir": dataset_info["data_dir"],
        "base_repo": dataset_info["base_repo"],
        "model_config": model_cfg.to_dict(),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "grad_clip": args.grad_clip,
        "precision": args.precision,
        "seed": args.seed,
        "objective": "CFM: t~Uniform(0,1), x_t=(1-t)x0+t*x1, target=x1-x0",
    }


def assert_resume_compatible(payload: dict, signature: dict) -> None:
    old = payload.get("train_signature")
    if old != signature:
        old_text = json.dumps(old, indent=2, sort_keys=True, default=str)
        new_text = json.dumps(signature, indent=2, sort_keys=True, default=str)
        raise RuntimeError(
            "Checkpoint training configuration does not match this launch.\n"
            f"--- checkpoint ---\n{old_text}\n--- requested ---\n{new_text}"
        )


def make_checkpoint_payload(
    *, model, ema, optimizer, step: int, args, model_cfg, dataset_info, signature,
    cumulative_elapsed_seconds: float,
) -> dict:
    return {
        "format": FORMAT,
        "step": int(step),
        "model": model.state_dict(),
        "ema_model": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model_cfg.to_dict(),
        "dataset_info": dataset_info,
        "train_signature": signature,
        "rng_state": capture_rng_state(),
        "cumulative_elapsed_seconds": float(cumulative_elapsed_seconds),
        "max_steps_at_save": int(args.max_steps),
    }


def append_metrics(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def fmt_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script.")
    if args.max_steps < 1 or args.batch_size < 1:
        raise ValueError("max-steps and batch-size must be positive")
    if args.autosave_every < 1 or args.log_every < 1:
        raise ValueError("autosave-every and log-every must be positive")
    if not (0 < args.ema_decay < 1):
        raise ValueError("ema-decay must be in (0,1)")

    run_dir = args.run_dir.expanduser().resolve()
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_time_axis_reference(run_dir / "time_axis_reference.csv")

    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print("[setup] loading unchanged full S1 through celeba80_cloud_bundle.CelebAStore ...", flush=True)
    reference, info_obj = load_full_s1(data_dir=args.data_dir, base_repo=args.base_repo, device=device)
    dataset_info = info_obj.__dict__.copy()
    print(
        f"[setup] full S1 loaded from paper_partitions.npz: "
        f"count={len(reference):,}, dimension={reference.shape[1]:,}, "
        f"resident={reference.nbytes/2**30:.2f} GiB",
        flush=True,
    )

    model_cfg = ModelConfig(args.base_channels, args.time_emb_dim, args.dropout)
    model = VelocityUNet80(model_cfg).to(device)
    ema = ModelEMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    signature = train_signature(args, model_cfg, dataset_info)

    uncompiled_model = model
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build has no torch.compile")
        model = torch.compile(model)
        print("[setup] torch.compile enabled", flush=True)

    start_step = 0
    previous_elapsed = 0.0
    if not args.no_auto_resume:
        resume_path, payload = find_best_resume(ckpt_dir)
        if payload is not None:
            assert_resume_compatible(payload, signature)
            uncompiled_model.load_state_dict(payload["model"])
            ema.load_state_dict(payload["ema_model"])
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            previous_elapsed = float(payload.get("cumulative_elapsed_seconds", 0.0))
            restore_rng_state(payload.get("rng_state"))
            print(f"[resume] automatically selected greatest valid checkpoint: {resume_path}", flush=True)
            print(f"[resume] continuing from step {start_step:,}", flush=True)

    if start_step >= args.max_steps:
        print(f"[done] checkpoint is already at step {start_step:,} >= max_steps={args.max_steps:,}; nothing to train.", flush=True)
        return

    config_payload = {
        "format": FORMAT,
        "code_dir_expected": "/root/shared-nvme/experiment-real-data/projects/celeba_NN_stage",
        "run_dir": str(run_dir),
        "official_time_axis": "t",
        "time_sampling": "Uniform(0,1)",
        "secondary_coordinate_only": "sigma=(1-t)/t",
        "path": "x_t=(1-t)*x0+t*x1",
        "source": "x0~N(0,I)",
        "target_data": f"x1~uniform(full S1, {len(reference):,})",
        "velocity_target": "x1-x0",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_info": dataset_info,
        "model_config": model_cfg.to_dict(),
        "train_signature": signature,
        "permanent_checkpoints": sorted(set(args.checkpoint_steps) | {args.max_steps}),
        "autosave": f"overwrite checkpoint_latest.pt every {args.autosave_every} steps",
    }
    atomic_json(config_payload, run_dir / "run_config.json")

    n_params = sum(p.numel() for p in uncompiled_model.parameters())
    print(f"[setup] model parameters={n_params:,}; EMA decay={args.ema_decay}", flush=True)
    print("[setup] PRIMARY TIME AXIS: t in [0,1], sampled uniformly. sigma is not used for training.", flush=True)
    print("[setup] secondary mapping only: sigma=(1-t)/t; see time_axis_reference.csv", flush=True)
    print(f"[setup] permanent checkpoints={sorted(set(args.checkpoint_steps) | {args.max_steps})}", flush=True)
    print(f"[setup] latest autosave every {args.autosave_every:,} steps; progress log every {args.log_every:,} steps", flush=True)

    milestone_steps = set(args.checkpoint_steps) | {args.max_steps}
    metrics_path = run_dir / "training_progress.csv"
    interval_loss = 0.0
    interval_count = 0
    session_start = time.time()
    interval_start = session_start
    autocast_enabled = args.precision == "bf16"

    model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        # Vanilla conditional flow matching, exactly paper-style time sampling.
        idx = torch.randint(0, len(reference), (args.batch_size,), device=device)
        x1 = reference[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device, dtype=torch.float32)  # Uniform [0,1)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        target_velocity = x1 - x0

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            pred_velocity = model(x_t, t)
            loss = F.mse_loss(pred_velocity, target_velocity)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(uncompiled_model.parameters(), args.grad_clip)
        optimizer.step()
        ema.update(uncompiled_model)

        interval_loss += float(loss.detach())
        interval_count += 1

        now = time.time()
        cumulative_elapsed = previous_elapsed + (now - session_start)
        do_autosave = step % args.autosave_every == 0 or step == args.max_steps
        do_milestone = step in milestone_steps
        if do_autosave or do_milestone:
            payload = make_checkpoint_payload(
                model=uncompiled_model, ema=ema, optimizer=optimizer, step=step,
                args=args, model_cfg=model_cfg, dataset_info=dataset_info,
                signature=signature, cumulative_elapsed_seconds=cumulative_elapsed,
            )
            if do_autosave:
                atomic_torch_save(payload, ckpt_dir / "checkpoint_latest.pt")
                atomic_json({"step": step, "path": "checkpoint_latest.pt"}, ckpt_dir / "latest_step.json")
            if do_milestone:
                milestone_path = ckpt_dir / f"checkpoint_step_{step:06d}.pt"
                atomic_torch_save(payload, milestone_path)
                print(f"[checkpoint] permanent {milestone_path.name}", flush=True)
            del payload

        if step % args.log_every == 0 or step == args.max_steps:
            interval_elapsed = max(now - interval_start, 1e-9)
            steps_per_sec = interval_count / interval_elapsed
            avg_loss = interval_loss / max(interval_count, 1)
            remaining = args.max_steps - step
            eta_sec = remaining / max(steps_per_sec, 1e-12)
            examples_seen = step * args.batch_size
            epochs_equiv = examples_seen / len(reference)
            peak_mem = torch.cuda.max_memory_allocated() / 2**30
            pct = 100.0 * step / args.max_steps
            row = {
                "step": step,
                "max_steps": args.max_steps,
                "progress_percent": f"{pct:.4f}",
                "avg_loss_last_interval": f"{avg_loss:.8f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8g}",
                "grad_norm_last": f"{float(grad_norm):.8f}",
                "examples_seen": examples_seen,
                "equivalent_epochs": f"{epochs_equiv:.6f}",
                "steps_per_second": f"{steps_per_sec:.6f}",
                "eta_seconds": f"{eta_sec:.3f}",
                "cumulative_elapsed_seconds": f"{cumulative_elapsed:.3f}",
                "peak_gpu_gib": f"{peak_mem:.4f}",
            }
            append_metrics(metrics_path, row)
            print(
                f"[progress] step={step:,}/{args.max_steps:,} ({pct:6.2f}%) | "
                f"loss={avg_loss:.6f} | epochs~{epochs_equiv:.2f} | "
                f"speed={steps_per_sec:.3f} step/s | ETA={fmt_duration(eta_sec)} | "
                f"elapsed={fmt_duration(cumulative_elapsed)} | peakGPU={peak_mem:.2f}GiB",
                flush=True,
            )
            interval_loss = 0.0
            interval_count = 0
            interval_start = now
            torch.cuda.reset_peak_memory_stats()

    print(f"[done] training reached step {args.max_steps:,}", flush=True)
    print(f"[done] run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
