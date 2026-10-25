#!/usr/bin/env python3
"""Theory-aligned numerical core for the CelebA80 empirical probability-flow ODE.

The model is the additive Gaussian family

    x_sigma = x_0 + sigma * epsilon, epsilon ~ N(0, I).

For the empirical data distribution, the exact conditional posterior is

    p(x_0=x_i | x_sigma=x) proportional to
    exp(-||x-x_i||^2 / (2 sigma^2)).

Tweedie's identity gives D_sigma(x)=E[x_0|x_sigma=x].  The VE
probability-flow ODE, using sigma itself as the independent variable, is

    dx / d sigma = (x - D_sigma(x)) / sigma.

Generation integrates from sigma_max down to sigma_min.  The same explicit
Gaussian sigma is used for the ODE score, saved stopping state, Tweedie
projection and ESS.  No residual-RMS surrogate is used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

FORMAT_NAME = "celeba80_gray_rgb_sum_shards_v1"
THEORY_FORMAT = "celeba80_explicit_gaussian_sigma_probability_flow_ode_v3"
ESS_DIRECT_FORMAT = "celeba80_direct_forward_gaussian_ess_v1"
PCA_ESS_DIRECT_FORMAT = "celeba80_pca_direct_forward_gaussian_ess_v1"
TRAJECTORY_FORMAT = "celeba80_explicit_sigma_fixed_ode_trajectory_v1"
EMPIRICAL_GENERATION_FORMAT = "celeba80_pca_empirical_generation_v1"
EMPIRICAL_GENERATION_TRAJECTORY_FORMAT = "celeba80_pca_empirical_generation_trajectory_v1"
EMPIRICAL_GENERATION_ESS_FORMAT = "celeba80_pca_empirical_generation_ess_distribution_v1"
EXPECTED_CLEAN_COUNT = 196_047
PAPER_HALF_SIZE = 98_023
PAPER_SUBSET_SIZES = (1, 10, 100, 1_000, 10_000, 98_023)
DISPLAY_SIZE = {PAPER_HALF_SIZE: PAPER_HALF_SIZE}
IMAGE_SHAPE = (80, 80)
DIMENSION = 80 * 80


def parse_csv_floats(text: str) -> Tuple[float, ...]:
    try:
        values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floating-point values.") from exc
    if not values or any((not math.isfinite(x)) or x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Sigma values must be finite and positive.")
    return tuple(dict.fromkeys(values))


def parse_csv_ints(text: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("Values must be positive integers.")
    return tuple(dict.fromkeys(values))


def sigma_tag(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_matmul(mode: str) -> None:
    allow_tf32 = mode == "tf32"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except (AttributeError, RuntimeError):
        pass


def atomic_json(path: Path, value: Mapping[str, object] | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def config_signature(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class CelebAStore:
    """Read the existing sharded CelebA80 prepared-data format."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared CelebA manifest not found: {manifest_path}\n"
                "Pass --data-dir as the directory that directly contains manifest.json."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: Dict[str, object] = json.load(handle)
        if self.manifest.get("format") != FORMAT_NAME:
            raise ValueError(f"Unsupported prepared-data format: {self.manifest.get('format')!r}")
        self.num_raw = int(self.manifest["num_raw_images"])
        self.storage = str(self.manifest["storage"])
        self.shards = list(self.manifest["shards"])
        self.decode_divisor = float(self.manifest["decode_divisor"])
        self.dim = DIMENSION

    def _load_shard(self, info: Mapping[str, object]) -> np.ndarray:
        path = self.data_dir / str(info["filename"])
        if self.storage == "npz_compressed":
            with np.load(path, allow_pickle=False) as archive:
                values = np.ascontiguousarray(archive["gray_rgb_sum"])
        elif self.storage == "npy":
            values = np.ascontiguousarray(np.load(path, mmap_mode="r", allow_pickle=False))
        else:
            raise ValueError(f"Unsupported shard storage: {self.storage}")
        expected = (int(info["count"]), *IMAGE_SHAPE)
        if values.shape != expected or values.dtype != np.uint16:
            raise ValueError(f"Bad shard {path}: got {values.shape}/{values.dtype}, expected {expected}/uint16.")
        return values

    def load_groups(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        partition_path = self.data_dir / "paper_partitions.npz"
        if not partition_path.is_file():
            raise FileNotFoundError(
                f"Missing {partition_path}. Use the existing preparation script before running experiments."
            )
        with np.load(partition_path, allow_pickle=False) as archive:
            indices_a = np.asarray(archive["group_a_raw_indices"], dtype=np.int64)
            indices_b = np.asarray(archive["group_b_raw_indices"], dtype=np.int64)
        raw = np.empty((self.num_raw, self.dim), dtype=np.uint16)
        cursor = 0
        for info in self.shards:
            images = self._load_shard(info)
            count = len(images)
            raw[cursor:cursor + count] = images.reshape(count, self.dim)
            cursor += count
        if cursor != self.num_raw:
            raise RuntimeError("Shard counts do not match manifest.")
        def transfer(indices: np.ndarray, label: str) -> torch.Tensor:
            selected = np.ascontiguousarray(raw[indices])
            tensor = torch.from_numpy(selected).to(device=device, dtype=torch.float32)
            tensor.mul_(1.0 / self.decode_divisor)
            print(f"resident group {label}: {tuple(tensor.shape)}, {tensor.nbytes / 2**30:.2f} GiB")
            return tensor
        group_a = transfer(indices_a, "S1")
        group_b = transfer(indices_b, "S2")
        del raw
        gc.collect()
        return group_a, group_b


def reference_norms(reference: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = torch.empty(reference.shape[0], device=reference.device, dtype=torch.float32)
    for start in range(0, len(reference), chunk_size):
        end = min(start + chunk_size, len(reference))
        out[start:end] = reference[start:end].square().sum(1)
    return out


def subset_for_group(
    group: str,
    full: torch.Tensor,
    full_norms: torch.Tensor,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n > len(full):
        raise ValueError(f"N={n:,} exceeds group size {len(full):,}.")
    if group in {"A", "S1"}:
        return full[:n], full_norms[:n]
    if group in {"B", "S2"}:
        return full[-n:], full_norms[-n:]
    raise ValueError(group)


@torch.inference_mode()
def sample_forward_gaussian_multi_sigma(
    reference: torch.Tensor,
    seeds: Sequence[int],
    sigmas: Sequence[float],
) -> Tuple[Dict[float, torch.Tensor], np.ndarray, torch.Tensor]:
    """Sample exact empirical forward marginals with a shared coupling.

    For each seed, choose one empirical center x_i and one epsilon ~ N(0, I),
    then return x_i + sigma * epsilon for every requested sigma.  Consequently,
    every fixed-sigma marginal is exactly

        (1/N) sum_i N(x_i, sigma^2 I),

    while the same center and epsilon are shared across sigma values.  This is
    ideal for ESS marginal-distribution experiments and avoids ODE integration.
    """
    if len(reference) < 1:
        raise ValueError("reference must be non-empty")
    sigma_values = tuple(float(s) for s in sigmas)
    if not sigma_values or any((not math.isfinite(s)) or s <= 0 for s in sigma_values):
        raise ValueError("sigmas must be finite and positive")
    centers = []
    noises = []
    center_indices = []
    for seed in seeds:
        generator = torch.Generator(device=reference.device).manual_seed(int(seed))
        center_index = int(
            torch.randint(0, len(reference), (1,), generator=generator, device=reference.device).item()
        )
        noise = torch.randn(
            reference.shape[1], generator=generator,
            device=reference.device, dtype=reference.dtype,
        )
        centers.append(reference[center_index])
        noises.append(noise)
        center_indices.append(center_index)
    center_batch = torch.stack(centers)
    noise_batch = torch.stack(noises)
    states = {sigma: center_batch + sigma * noise_batch for sigma in sigma_values}
    return states, np.asarray(center_indices, dtype=np.int64), noise_batch


@torch.inference_mode()
def empirical_posterior(
    queries: torch.Tensor,
    reference: torch.Tensor,
    norms: torch.Tensor,
    sigma: torch.Tensor | float,
    chunk_size: int,
    topk: int = 2,
) -> Dict[str, torch.Tensor]:
    """Exact streaming Gaussian posterior over every reference center."""
    if queries.ndim != 2 or reference.ndim != 2 or queries.shape[1] != reference.shape[1]:
        raise ValueError("queries and reference must be [batch, dimension].")
    if topk < 1:
        raise ValueError("topk must be positive.")
    batch = queries.shape[0]
    sigma_t = torch.as_tensor(sigma, dtype=queries.dtype, device=queries.device)
    if sigma_t.ndim == 0:
        sigma_t = sigma_t.expand(batch)
    if sigma_t.shape != (batch,) or bool((sigma_t <= 0).any()):
        raise ValueError("sigma must be positive and scalar or shape [batch].")
    query_norm = queries.square().sum(1)
    running_max = torch.full((batch,), -torch.inf, device=queries.device, dtype=queries.dtype)
    denominator = torch.zeros(batch, device=queries.device, dtype=queries.dtype)
    squared_denominator = torch.zeros_like(denominator)
    numerator = torch.zeros_like(queries)
    best_values = torch.full((batch, topk), -torch.inf, device=queries.device, dtype=queries.dtype)
    best_indices = torch.full((batch, topk), -1, device=queries.device, dtype=torch.long)
    inv_two_sigma2 = 0.5 / sigma_t.square()
    for start in range(0, len(reference), chunk_size):
        end = min(start + chunk_size, len(reference))
        ref = reference[start:end]
        distances = query_norm[:, None] + norms[start:end][None, :] - 2.0 * (queries @ ref.T)
        distances.clamp_min_(0.0)
        logits = -distances * inv_two_sigma2[:, None]
        local_max = logits.amax(1)
        new_max = torch.maximum(running_max, local_max)
        old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(logits - new_max[:, None])
        denominator.mul_(old_scale).add_(weights.sum(1))
        squared_denominator.mul_(old_scale.square()).add_(weights.square().sum(1))
        numerator.mul_(old_scale[:, None]).add_(weights @ ref)
        running_max = new_max
        local_k = min(topk, end - start)
        local_values, local_indices = torch.topk(logits, local_k, dim=1)
        local_indices += start
        if local_k < topk:
            pad = topk - local_k
            local_values = torch.cat([local_values, torch.full((batch, pad), -torch.inf, device=queries.device)], 1)
            local_indices = torch.cat([local_indices, torch.full((batch, pad), -1, device=queries.device, dtype=torch.long)], 1)
        merged_values = torch.cat([best_values, local_values], 1)
        merged_indices = torch.cat([best_indices, local_indices], 1)
        best_values, positions = torch.topk(merged_values, topk, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    tiny = torch.finfo(queries.dtype).tiny
    mean = numerator / denominator[:, None].clamp_min(tiny)
    ess = denominator.square() / squared_denominator.clamp_min(tiny)
    top_probabilities = (
        torch.exp(best_values - running_max[:, None])
        / denominator[:, None].clamp_min(tiny)
    )
    top_probability = top_probabilities[:, 0]
    return {
        "mean": mean,
        "ess": ess,
        "top_probability": top_probability,
        "topk_probability_mass": top_probabilities.sum(1),
        "top_indices": best_indices,
        "top_logits": best_values,
        "log_normalizer": running_max + torch.log(denominator.clamp_min(tiny)),
    }


@torch.inference_mode()
def posterior_batched(
    queries: torch.Tensor,
    reference: torch.Tensor,
    norms: torch.Tensor,
    sigma: torch.Tensor | float,
    query_batch_size: int,
    reference_chunk_size: int,
    topk: int = 2,
) -> Dict[str, torch.Tensor]:
    pieces: Dict[str, List[torch.Tensor]] = {}
    sigma_t = torch.as_tensor(sigma, device=queries.device, dtype=queries.dtype)
    for start in range(0, len(queries), query_batch_size):
        end = min(start + query_batch_size, len(queries))
        local_sigma = sigma_t if sigma_t.ndim == 0 else sigma_t[start:end]
        result = empirical_posterior(
            queries[start:end], reference, norms, local_sigma, reference_chunk_size, topk=topk
        )
        for key, value in result.items():
            pieces.setdefault(key, []).append(value)
    return {key: torch.cat(values, 0) for key, values in pieces.items()}


def karras_sigma_schedule(
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float = 7.0,
) -> np.ndarray:
    if not (sigma_max > sigma_min > 0):
        raise ValueError("Require sigma_max > sigma_min > 0.")
    if steps < 2 or rho <= 0:
        raise ValueError("steps must be >=2 and rho positive.")
    ramp = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    inv_rho_max = sigma_max ** (1.0 / rho)
    inv_rho_min = sigma_min ** (1.0 / rho)
    values = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho
    values[0] = sigma_max
    values[-1] = sigma_min
    return values


def merged_sigma_schedule(
    sigma_max: float,
    sigma_min: float,
    steps: int,
    targets: Sequence[float],
    rho: float = 7.0,
) -> np.ndarray:
    targets_array = np.asarray(targets, dtype=np.float64)
    if np.any(targets_array > sigma_max + 1e-12) or np.any(targets_array < sigma_min - 1e-12):
        raise ValueError(
            f"Every target sigma must lie in [{sigma_min:g}, {sigma_max:g}]."
        )
    base = karras_sigma_schedule(sigma_max, sigma_min, steps, rho)
    merged = np.unique(np.concatenate([base, targets_array, [sigma_max, sigma_min]]))
    return np.sort(merged)[::-1]


def normalized_log_time(sigmas: np.ndarray, sigma_max: float, sigma_min: float) -> np.ndarray:
    values = np.asarray(sigmas, dtype=np.float64)
    denominator = math.log(sigma_max) - math.log(sigma_min)
    return (math.log(sigma_max) - np.log(values)) / denominator


def _seed_initial_state(
    reference: torch.Tensor,
    mean: torch.Tensor,
    seeds: Sequence[int],
    sigma_max: float,
    initialization: str,
) -> Tuple[torch.Tensor, np.ndarray]:
    device = reference.device
    states = []
    center_indices = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        if initialization == "empirical_mixture":
            center_index = int(
                torch.randint(0, len(reference), (1,), generator=generator, device=device).item()
            )
            center = reference[center_index]
        elif initialization == "gaussian_mean":
            center_index = -1
            center = mean
        elif initialization == "gaussian_noise":
            center_index = -1
            center = torch.zeros_like(mean)
        else:
            raise ValueError(initialization)
        noise = torch.randn(
            reference.shape[1], generator=generator, device=device, dtype=reference.dtype
        )
        states.append(center + float(sigma_max) * noise)
        center_indices.append(center_index)
    return torch.stack(states), np.asarray(center_indices, dtype=np.int64)


def gaussian_noise_initial_state(
    dimension: int,
    seeds: Sequence[int],
    sigma_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the unconditional VE terminal state ``sigma_max * epsilon``.

    The caller can construct this tensor once and pass the exact same tensor to
    two empirical flows.  That avoids relying only on matching RNG calls when
    comparing S1 and S2.
    """
    if dimension < 1 or not math.isfinite(float(sigma_max)) or sigma_max <= 0:
        raise ValueError("dimension and sigma_max must be positive.")
    states = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(dimension, generator=generator, device=device, dtype=dtype)
        states.append(float(sigma_max) * noise)
    if not states:
        raise ValueError("seeds must be non-empty.")
    return torch.stack(states)


@torch.inference_mode()
def probability_flow_trajectory(
    reference: torch.Tensor,
    norms: torch.Tensor,
    mean: torch.Tensor,
    seeds: Sequence[int],
    target_sigmas: Sequence[float],
    *,
    sigma_max: float,
    sigma_min: float,
    ode_steps: int,
    rho: float,
    solver: str,
    initialization: str,
    query_batch_size: int,
    reference_chunk_size: int,
    log_frequency: int = 0,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[Dict[float, torch.Tensor], Dict[float, Dict[str, np.ndarray]]]:
    """Integrate the exact empirical VE probability-flow ODE.

    Target states are saved at exact schedule values, not at a residual crossing.
    """
    targets = tuple(sorted(set(float(x) for x in target_sigmas), reverse=True))
    schedule = merged_sigma_schedule(sigma_max, sigma_min, ode_steps, targets, rho)
    if initial_state is None:
        y, center_indices = _seed_initial_state(reference, mean, seeds, sigma_max, initialization)
    else:
        expected = (len(seeds), reference.shape[1])
        if tuple(initial_state.shape) != expected:
            raise ValueError(f"initial_state has shape {tuple(initial_state.shape)}, expected {expected}.")
        y = initial_state.to(device=reference.device, dtype=reference.dtype).clone()
        center_indices = np.full(len(seeds), -1, dtype=np.int64)
    states: Dict[float, torch.Tensor] = {}
    diagnostics: Dict[float, Dict[str, np.ndarray]] = {}

    def maybe_capture(sigma_value: float, step_index: int) -> None:
        for target in targets:
            if target not in states and math.isclose(sigma_value, target, rel_tol=0.0, abs_tol=1e-11):
                states[target] = y.clone()
                diagnostics[target] = {
                    "capture_sigma": np.full(len(seeds), target, dtype=np.float64),
                    "capture_ode_step": np.full(len(seeds), step_index, dtype=np.int32),
                    "initial_center_index": center_indices.copy(),
                }

    maybe_capture(float(schedule[0]), 0)
    for step_index, (sigma_current, sigma_next) in enumerate(zip(schedule[:-1], schedule[1:]), 1):
        sigma_current = float(sigma_current)
        sigma_next = float(sigma_next)
        current_result = posterior_batched(
            y, reference, norms, sigma_current, query_batch_size, reference_chunk_size
        )
        drift_current = (y - current_result["mean"]) / sigma_current
        delta = sigma_next - sigma_current
        if solver == "euler":
            y = y + delta * drift_current
        elif solver == "heun":
            predicted = y + delta * drift_current
            next_result = posterior_batched(
                predicted, reference, norms, sigma_next, query_batch_size, reference_chunk_size
            )
            drift_next = (predicted - next_result["mean"]) / sigma_next
            y = y + 0.5 * delta * (drift_current + drift_next)
        else:
            raise ValueError(solver)
        maybe_capture(sigma_next, step_index)
        if log_frequency and (step_index == 1 or step_index % log_frequency == 0):
            print(
                f"ODE step {step_index:4d}/{len(schedule)-1}: "
                f"sigma {sigma_current:.6g} -> {sigma_next:.6g}"
            )

    missing = [target for target in targets if target not in states]
    if missing:
        raise RuntimeError(f"Failed to capture exact sigma targets: {missing}")
    return states, diagnostics


@torch.inference_mode()
def project_at_sigma(
    state: torch.Tensor,
    reference: torch.Tensor,
    norms: torch.Tensor,
    sigma: float,
    query_batch_size: int,
    reference_chunk_size: int,
    topk: int = 2,
) -> Dict[str, torch.Tensor]:
    """Tweedie projection and ESS at the same Gaussian sigma as the state."""
    return posterior_batched(
        state,
        reference,
        norms,
        float(sigma),
        query_batch_size,
        reference_chunk_size,
        topk=topk,
    )


@torch.inference_mode()
def closest_centered_cosine(
    queries: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = queries - queries.mean(1, keepdim=True)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-6)
    best_value = torch.full((len(queries),), -torch.inf, device=queries.device)
    best_index = torch.full((len(queries),), -1, dtype=torch.long, device=queries.device)
    for start in range(0, len(reference), chunk_size):
        end = min(start + chunk_size, len(reference))
        ref = reference[start:end]
        ref = ref - ref.mean(1, keepdim=True)
        ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-6)
        value, index = (q @ ref.T).max(1)
        update = value > best_value
        best_value[update] = value[update]
        best_index[update] = index[update] + start
    return best_value.clamp(-1, 1), best_index


def paired_centered_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a0 = a - a.mean(1, keepdim=True)
    b0 = b - b.mean(1, keepdim=True)
    return (
        (a0 * b0).sum(1)
        / (a0.norm(dim=1).clamp_min(1e-6) * b0.norm(dim=1).clamp_min(1e-6))
    ).clamp(-1, 1)


def image_sharpness(images: torch.Tensor) -> torch.Tensor:
    x = images.reshape(-1, *IMAGE_SHAPE)
    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = x[:, 1:, :] - x[:, :-1, :]
    return torch.sqrt(0.5 * (dx.square().mean((1, 2)) + dy.square().mean((1, 2))))


def select_representative_indices(similarity: np.ndarray, count: int) -> Dict[str, int]:
    values = np.asarray(similarity, dtype=np.float64)
    if values.size == 0:
        raise ValueError("No samples.")
    fixed = 0
    best = int(np.nanargmax(values))
    median = float(np.nanmedian(values))
    median_index = int(np.nanargmin(np.abs(values - median)))
    return {
        "fixed_seed": min(fixed, count - 1),
        "best_similarity": best,
        "median_similarity": median_index,
    }


def validate_theory_archive(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        format_value = str(np.asarray(archive["format"]).reshape(()).item()) if "format" in archive else ""
    if format_value != THEORY_FORMAT:
        raise ValueError(
            f"{path} is a legacy/non-aligned result ({format_value!r}). "
            "It cannot be reused by the explicit-sigma implementation."
        )
