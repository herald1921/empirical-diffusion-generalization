#!/usr/bin/env python3
"""Shared helpers for unconditional PCA empirical-diffusion experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

import celeba80_theory_core as core
import pca_theory_utils as pca


def transform_data(data: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "spatial":
        return data
    if representation == "frequency":
        return pca.rfft_features(data)
    raise ValueError(representation)


def inverse_data(features: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "spatial":
        return features
    if representation == "frequency":
        return pca.inverse_rfft_features(features)
    raise ValueError(representation)


def _balanced_fit_pixels(
    group_s1: torch.Tensor,
    group_s2: torch.Tensor,
    fit_samples: int,
) -> torch.Tensor:
    """Select a deterministic, balanced S1/S2 sample without concatenating all data."""
    if fit_samples < 2:
        raise ValueError("pca_fit_samples must be at least 2.")
    count_s1 = min(len(group_s1), (fit_samples + 1) // 2)
    count_s2 = min(len(group_s2), fit_samples - count_s1)
    if count_s2 < 1:
        count_s2 = 1
        count_s1 = min(len(group_s1), fit_samples - 1)
    idx_s1 = torch.from_numpy(pca.evenly_spaced_indices(len(group_s1), count_s1)).to(group_s1.device)
    idx_s2 = torch.from_numpy(pca.evenly_spaced_indices(len(group_s2), count_s2)).to(group_s2.device)
    return torch.cat([group_s1[idx_s1], group_s2[idx_s2]], dim=0)


def fit_or_load_shared_pca(
    group_s1: torch.Tensor,
    group_s2: torch.Tensor,
    *,
    representation: str,
    rank: int,
    fit_samples: int,
    oversample: int,
    power_iterations: int,
    seed: int,
    cache_path: Path,
    overwrite_cache: bool,
) -> pca.PCAModel:
    """Fit one PCA basis shared by S1 and S2, or load the matching cache."""
    if cache_path.is_file() and not overwrite_cache:
        model = pca.load_model(cache_path, group_s1.device)
        if model.components.shape[0] < rank:
            raise ValueError(
                f"Cached PCA has rank {model.components.shape[0]}, requested {rank}. "
                "Use --overwrite-cache or a different --cache-dir."
            )
        return model

    fit_pixels = _balanced_fit_pixels(group_s1, group_s2, fit_samples)
    fit_representation = transform_data(fit_pixels, representation)
    max_rank = min(rank, fit_representation.shape[0] - 1, fit_representation.shape[1])
    if max_rank != rank:
        raise ValueError(
            f"Cannot fit rank={rank} from {len(fit_representation)} fit samples and "
            f"dimension={fit_representation.shape[1]}. Increase --pca-fit-samples."
        )
    model = pca.fit_randomized_pca(
        fit_representation,
        rank,
        fit_samples=len(fit_representation),
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )
    pca.save_model(cache_path, model, representation)
    return model


def prepare_latent_references(
    group_s1: torch.Tensor,
    group_s2: torch.Tensor,
    model: pca.PCAModel,
    *,
    representation: str,
    rank: int,
    subset_size: int,
    reference_chunk_size: int,
) -> Dict[str, torch.Tensor]:
    """Create disjoint S1/S2 pixel and PCA-latent reference sets."""
    if subset_size < 1 or subset_size > min(len(group_s1), len(group_s2)):
        raise ValueError(
            f"subset_size={subset_size} exceeds available S1/S2 size "
            f"({len(group_s1)}/{len(group_s2)})."
        )
    pixel_s1 = group_s1[:subset_size]
    pixel_s2 = group_s2[-subset_size:]
    latent_s1 = pca.encode(transform_data(pixel_s1, representation), model, rank)
    latent_s2 = pca.encode(transform_data(pixel_s2, representation), model, rank)
    return {
        "pixel_s1": pixel_s1,
        "pixel_s2": pixel_s2,
        "latent_s1": latent_s1,
        "latent_s2": latent_s2,
        "norm_s1": core.reference_norms(latent_s1, reference_chunk_size),
        "norm_s2": core.reference_norms(latent_s2, reference_chunk_size),
    }


def decode_clean_latent(
    clean_latent: torch.Tensor,
    model: pca.PCAModel,
    *,
    representation: str,
    rank: int,
    method: str,
    sigma: float,
    shrinkage_strength: float,
) -> torch.Tensor:
    if method == "hard":
        reconstructed = pca.decode(clean_latent, model, rank)
    elif method == "shrinkage":
        reconstructed = pca.wiener_decode(
            clean_latent, model, rank, sigma, shrinkage_strength
        )
    else:
        raise ValueError(method)
    return inverse_data(reconstructed, representation).clamp(0, 1)


def decode_state_latent(
    state: torch.Tensor,
    model: pca.PCAModel,
    *,
    representation: str,
    rank: int,
) -> torch.Tensor:
    return inverse_data(pca.decode(state, model, rank), representation).clamp(0, 1)


def same_pca_operation(
    images: torch.Tensor,
    model: pca.PCAModel,
    *,
    representation: str,
    rank: int,
    method: str,
    sigma: float,
    shrinkage_strength: float,
) -> torch.Tensor:
    latent = pca.encode(transform_data(images, representation), model, rank)
    return decode_clean_latent(
        latent,
        model,
        representation=representation,
        rank=rank,
        method=method,
        sigma=sigma,
        shrinkage_strength=shrinkage_strength,
    )


@torch.inference_mode()
def generate_pair(
    refs: Dict[str, torch.Tensor],
    initial_state: torch.Tensor,
    seeds: Sequence[int],
    *,
    sigma_max: float,
    sigma_stop: float,
    ode_steps: int,
    rho: float,
    solver: str,
    query_batch_size: int,
    reference_chunk_size: int,
    posterior_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run S1 and S2 from one exact shared Gaussian terminal state."""
    common = dict(
        target_sigmas=(sigma_stop,),
        sigma_max=sigma_max,
        sigma_min=sigma_stop,
        ode_steps=ode_steps,
        rho=rho,
        solver=solver,
        initialization="gaussian_noise",
        query_batch_size=query_batch_size,
        reference_chunk_size=reference_chunk_size,
        initial_state=initial_state,
    )
    states_s1, _ = core.probability_flow_trajectory(
        refs["latent_s1"], refs["norm_s1"], refs["latent_s1"].mean(0), seeds, **common
    )
    states_s2, _ = core.probability_flow_trajectory(
        refs["latent_s2"], refs["norm_s2"], refs["latent_s2"].mean(0), seeds, **common
    )
    state_s1 = states_s1[float(sigma_stop)]
    state_s2 = states_s2[float(sigma_stop)]
    post_s1 = core.project_at_sigma(
        state_s1,
        refs["latent_s1"],
        refs["norm_s1"],
        sigma_stop,
        query_batch_size,
        reference_chunk_size,
        topk=posterior_topk,
    )
    post_s2 = core.project_at_sigma(
        state_s2,
        refs["latent_s2"],
        refs["norm_s2"],
        sigma_stop,
        query_batch_size,
        reference_chunk_size,
        topk=posterior_topk,
    )
    return state_s1, state_s2, post_s1, post_s2


def representative_indices(similarity: np.ndarray, ess_s1: np.ndarray, ess_s2: np.ndarray) -> Dict[str, int]:
    pair_ess = 0.5 * (np.asarray(ess_s1) + np.asarray(ess_s2))
    median = float(np.nanmedian(pair_ess))
    return {
        "fixed_seed": 0,
        "best_cross_similarity": int(np.nanargmax(similarity)),
        "median_ess": int(np.nanargmin(np.abs(pair_ess - median))),
    }
