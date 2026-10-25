"""PCA and frequency-transform utilities for explicit-sigma experiments."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple
import numpy as np
import torch
import celeba80_theory_core as core


@dataclass
class PCAModel:
    mean: torch.Tensor
    components: torch.Tensor   # [rank, D]
    eigenvalues: torch.Tensor  # [rank]


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    count = min(length, count)
    return np.linspace(0, length - 1, count, dtype=np.int64)


@torch.inference_mode()
def fit_randomized_pca(
    data: torch.Tensor,
    rank: int,
    *,
    fit_samples: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> PCAModel:
    indices = evenly_spaced_indices(len(data), fit_samples)
    x = data[torch.from_numpy(indices).to(data.device)]
    mean = x.mean(0)
    centered = x - mean
    q = min(rank + oversample, centered.shape[0], centered.shape[1])
    generator = torch.Generator(device=data.device).manual_seed(seed)
    omega = torch.randn(
        centered.shape[1], q, generator=generator, device=data.device, dtype=data.dtype
    )
    y = centered @ omega
    for _ in range(power_iterations):
        y = centered @ (centered.T @ y)
    qmat = torch.linalg.qr(y, mode="reduced").Q
    b = qmat.T @ centered
    _, singular, vh = torch.linalg.svd(b, full_matrices=False)
    components = vh[:rank].contiguous()
    eigenvalues = singular[:rank].square() / max(1, centered.shape[0] - 1)
    return PCAModel(mean=mean, components=components, eigenvalues=eigenvalues)


def encode(data: torch.Tensor, model: PCAModel, rank: int) -> torch.Tensor:
    return (data - model.mean) @ model.components[:rank].T


def decode(scores: torch.Tensor, model: PCAModel, rank: int) -> torch.Tensor:
    return model.mean + scores @ model.components[:rank]


def wiener_decode(
    clean_score_estimate: torch.Tensor,
    model: PCAModel,
    rank: int,
    sigma: float,
    shrinkage_strength: float,
) -> torch.Tensor:
    lam = model.eigenvalues[:rank].clamp_min(1e-12)
    alpha = lam / (lam + shrinkage_strength * float(sigma) ** 2)
    return model.mean + (clean_score_estimate * alpha) @ model.components[:rank]


def rfft_features(images_flat: torch.Tensor) -> torch.Tensor:
    images = images_flat.reshape(-1, *core.IMAGE_SHAPE)
    spectrum = torch.fft.rfft2(images, norm="ortho")
    weights = torch.ones(spectrum.shape[-1], device=images.device, dtype=images.dtype)
    if spectrum.shape[-1] > 2:
        weights[1:-1] = math.sqrt(2.0)
    weighted = spectrum * weights[None, None, :]
    return torch.cat([weighted.real.reshape(len(images), -1), weighted.imag.reshape(len(images), -1)], 1)


def inverse_rfft_features(features: torch.Tensor) -> torch.Tensor:
    h, w = core.IMAGE_SHAPE
    fw = w // 2 + 1
    half = h * fw
    real = features[:, :half].reshape(-1, h, fw)
    imag = features[:, half:].reshape(-1, h, fw)
    weights = torch.ones(fw, device=features.device, dtype=features.dtype)
    if fw > 2:
        weights[1:-1] = math.sqrt(2.0)
    spectrum = torch.complex(real, imag) / weights[None, None, :]
    images = torch.fft.irfft2(spectrum, s=(h, w), norm="ortho")
    return images.reshape(len(features), -1)


def save_model(path: Path, model: PCAModel, representation: str) -> None:
    core.atomic_npz(
        path,
        format=np.asarray(core.THEORY_FORMAT),
        representation=np.asarray(representation),
        mean=model.mean.cpu().numpy(),
        components=model.components.cpu().numpy(),
        eigenvalues=model.eigenvalues.cpu().numpy(),
    )


def load_model(path: Path, device: torch.device) -> PCAModel:
    with np.load(path, allow_pickle=False) as z:
        if str(z["format"].item()) != core.THEORY_FORMAT:
            raise ValueError(f"Legacy PCA cache rejected: {path}")
        return PCAModel(
            mean=torch.from_numpy(z["mean"]).to(device),
            components=torch.from_numpy(z["components"]).to(device),
            eigenvalues=torch.from_numpy(z["eigenvalues"]).to(device),
        )
