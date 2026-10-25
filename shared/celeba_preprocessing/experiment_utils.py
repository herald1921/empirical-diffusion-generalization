#!/usr/bin/env python3
"""Shared PCA, metric, and compact-result helpers for CelebA80 experiments."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch

import celeba80_gallery_pairing as gallery_pairing
import run_celeba80_empirical_oracle as base


def parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip() for item in text.split(",") if item.strip()))
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_pca(
    reference: torch.Tensor,
    max_rank: int,
    *,
    oversample: int,
    niter: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    mean = reference.mean(dim=0)
    rank = min(max_rank, reference.shape[0] - 1, reference.shape[1])
    if rank < 1:
        return {
            "mean": mean,
            "components": torch.empty((reference.shape[1], 0), device=reference.device),
            "eigenvalues": torch.empty((0,), device=reference.device),
        }
    q = min(rank + oversample, reference.shape[0], reference.shape[1])
    torch.manual_seed(seed)
    if reference.is_cuda:
        torch.cuda.manual_seed_all(seed)
    _, singular_values, components = torch.pca_lowrank(
        reference - mean, q=q, center=False, niter=niter
    )
    components = components[:, :rank].contiguous()
    eigenvalues = singular_values[:rank].square() / max(reference.shape[0] - 1, 1)
    return {"mean": mean, "components": components, "eigenvalues": eigenvalues}


def encode_pca(values: torch.Tensor, model: Mapping[str, torch.Tensor], rank: int) -> torch.Tensor:
    components = model["components"][:, :rank]
    return (values - model["mean"]) @ components


def decode_pca(scores: torch.Tensor, model: Mapping[str, torch.Tensor], rank: int) -> torch.Tensor:
    components = model["components"][:, :rank]
    return model["mean"] + scores @ components.T


def pca_project(
    values: torch.Tensor,
    model: Mapping[str, torch.Tensor],
    rank: int,
    *,
    method: str,
    noise_scale: float,
    clip: bool = True,
) -> torch.Tensor:
    scores = encode_pca(values, model, rank)
    if method == "shrinkage":
        eigenvalues = model["eigenvalues"][:rank]
        gains = eigenvalues / (eigenvalues + noise_scale**2)
        scores = scores * gains
    elif method != "hard":
        raise ValueError(f"Unknown PCA method: {method}")
    result = decode_pca(scores, model, rank)
    return result.clamp(0, 1) if clip else result


def sharpness(images: torch.Tensor) -> torch.Tensor:
    values = images.reshape(-1, 80, 80)
    dx = values[:, :, 1:] - values[:, :, :-1]
    dy = values[:, 1:, :] - values[:, :-1, :]
    return torch.sqrt(0.5 * (dx.square().mean((1, 2)) + dy.square().mean((1, 2))))


@torch.inference_mode()
def metric_bundle(
    sample_a: torch.Tensor,
    sample_b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    reference_chunk_size: int,
) -> Dict[str, np.ndarray]:
    nearest_a, index_a = base.closest_centered_cosine(sample_a, reference_a, reference_chunk_size)
    nearest_b, index_b = base.closest_centered_cosine(sample_b, reference_b, reference_chunk_size)
    return {
        "similarity_ab": base.paired_centered_cosine(sample_a, sample_b).cpu().numpy(),
        "nearest_similarity_a": nearest_a.cpu().numpy(),
        "nearest_similarity_b": nearest_b.cpu().numpy(),
        "nearest_index_a": index_a.cpu().numpy(),
        "nearest_index_b": index_b.cpu().numpy(),
        "sharpness_a": sharpness(sample_a).cpu().numpy(),
        "sharpness_b": sharpness(sample_b).cpu().numpy(),
    }


def compact_payload(
    sample_a: torch.Tensor,
    sample_b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    metrics: Mapping[str, np.ndarray],
    *,
    prefix: str = "",
    gallery_count: int = 8,
    gallery_index: int = 0,
    max_similarity_elements: int = gallery_pairing.DEFAULT_MAX_SIMILARITY_ELEMENTS,
    save_all_samples: bool = False,
) -> MutableMapping[str, np.ndarray]:
    a = sample_a.detach().cpu().numpy().astype(np.float32, copy=False)
    b = sample_b.detach().cpu().numpy().astype(np.float32, copy=False)
    nearest_a = np.asarray(metrics["nearest_index_a"], dtype=np.int64)
    nearest_b = np.asarray(metrics["nearest_index_b"], dtype=np.int64)
    payload: MutableMapping[str, np.ndarray] = {}
    stem = f"{prefix}_" if prefix else ""
    for key, value in metrics.items():
        payload[f"{stem}{key}"] = np.asarray(value)
    payload.update(gallery_pairing.build_best_paired_payload(
        a, b, nearest_a, nearest_b, reference_a, reference_b,
        prefix=prefix, block_size=max_similarity_elements,
    ))
    payload.update(gallery_pairing.build_median_paired_payload(
        a, b, nearest_a, nearest_b, reference_a, reference_b,
        prefix=prefix, block_size=max_similarity_elements,
    ))
    payload.update(gallery_pairing.build_best_cross_payload(
        a, b, nearest_a, nearest_b, reference_a, reference_b,
        prefix=prefix, max_similarity_elements=max_similarity_elements,
    ))
    count = min(gallery_count, len(a))
    payload[f"{stem}sample_a"] = np.asarray(a[:count])
    payload[f"{stem}sample_b"] = np.asarray(b[:count])
    payload[f"{stem}gallery_indices"] = np.arange(count, dtype=np.int64)
    payload[f"{stem}nearest_train_a"] = reference_a[torch.as_tensor(nearest_a[:count], device=reference_a.device)].cpu().numpy()
    payload[f"{stem}nearest_train_b"] = reference_b[torch.as_tensor(nearest_b[:count], device=reference_b.device)].cpu().numpy()
    fixed = min(max(gallery_index, 0), len(a) - 1)
    payload[f"{stem}fixed_index"] = np.asarray(fixed, dtype=np.int64)
    if save_all_samples:
        payload[f"{stem}all_sample_a"] = np.asarray(a)
        payload[f"{stem}all_sample_b"] = np.asarray(b)
    return payload


def summary_metrics(metrics: Mapping[str, np.ndarray]) -> Dict[str, float]:
    similarity = np.asarray(metrics["similarity_ab"], dtype=np.float64)
    nearest = np.concatenate((metrics["nearest_similarity_a"], metrics["nearest_similarity_b"])).astype(np.float64)
    sharp = np.concatenate((metrics["sharpness_a"], metrics["sharpness_b"])).astype(np.float64)
    return {
        "mean_similarity_between_models": float(np.mean(similarity)),
        "median_similarity_between_models": float(np.median(similarity)),
        "mean_similarity_to_closest_train": float(np.mean(nearest)),
        "median_similarity_to_closest_train": float(np.median(nearest)),
        "mean_sharpness": float(np.mean(sharp)),
        "median_sharpness": float(np.median(sharp)),
    }
