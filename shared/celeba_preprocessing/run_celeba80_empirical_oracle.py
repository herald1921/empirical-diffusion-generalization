#!/usr/bin/env python3
"""Compatibility entry point and simple explicit-sigma empirical ODE generator."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from celeba80_theory_core import *
from project_paths import CELEBA80_DATA_DIR, ORACLE_RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CELEBA80_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ORACLE_RESULTS_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--generation-samples", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--stop-sigmas", type=parse_csv_floats, default=(8, 4, 2, 1, .5, .2, .1))
    parser.add_argument("--sigma-max", type=float, default=16.0)
    parser.add_argument("--sigma-min", type=float, default=.1)
    parser.add_argument("--ode-steps", type=int, default=80)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--initialization", choices=("empirical_mixture", "gaussian_mean", "gaussian_noise"), default="empirical_mixture")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    parser.add_argument("--matmul", choices=("highest", "tf32"), default="highest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)
    configure_matmul(args.matmul)
    seed_everything(args.seed)
    store = CelebAStore(args.data_dir)
    group_a, group_b = store.load_groups(device)
    norms_a = reference_norms(group_a, args.reference_chunk_size)
    norms_b = reference_norms(group_b, args.reference_chunk_size)
    ref_a, norm_a = subset_for_group("S1", group_a, norms_a, args.subset_size)
    ref_b, norm_b = subset_for_group("S2", group_b, norms_b, args.subset_size)
    rows = {}
    for flow, ref, norms in (("S1", ref_a, norm_a), ("S2", ref_b, norm_b)):
        chunks = {sigma: [] for sigma in args.stop_sigmas}
        for start in range(0, args.generation_samples, args.generation_batch_size):
            end = min(start + args.generation_batch_size, args.generation_samples)
            seeds = range(args.seed + start, args.seed + end)
            states, _ = probability_flow_trajectory(
                ref, norms, ref.mean(0), seeds, args.stop_sigmas,
                sigma_max=args.sigma_max, sigma_min=args.sigma_min,
                ode_steps=args.ode_steps, rho=args.rho, solver=args.solver,
                initialization=args.initialization,
                query_batch_size=args.query_batch_size,
                reference_chunk_size=args.reference_chunk_size,
                log_frequency=0,
            )
            for sigma in args.stop_sigmas:
                chunks[sigma].append(project_at_sigma(
                    states[sigma], ref, norms, sigma,
                    args.query_batch_size, args.reference_chunk_size
                )["mean"].cpu().numpy())
        rows[flow] = {sigma: np.concatenate(parts) for sigma, parts in chunks.items()}
    output = args.output_dir.expanduser().resolve()
    arrays = {
        "format": np.asarray(THEORY_FORMAT),
        "stop_sigmas": np.asarray(args.stop_sigmas, dtype=np.float64),
        "sigma_max": np.asarray(args.sigma_max),
        "sigma_min": np.asarray(args.sigma_min),
    }
    for flow in ("S1", "S2"):
        for sigma in args.stop_sigmas:
            arrays[f"{flow}_sigma_{sigma_tag(sigma)}"] = rows[flow][sigma]
    atomic_npz(output / "explicit_sigma_generation.npz", **arrays)
    print(output / "explicit_sigma_generation.npz")


if __name__ == "__main__":
    main()
