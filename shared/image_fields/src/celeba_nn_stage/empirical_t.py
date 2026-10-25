from __future__ import annotations

import torch


def sigma_from_t(t: float) -> float:
    t = float(t)
    if not (0.0 < t < 1.0):
        raise ValueError("empirical field conversion requires 0 < t < 1")
    return (1.0 - t) / t


def empirical_velocity_from_mean(x_t: torch.Tensor, posterior_mean_x1: torch.Tensor, t: float) -> torch.Tensor:
    """Closed-form FM empirical velocity u*(x,t)=(E[x1|x_t]-x_t)/(1-t)."""
    t = float(t)
    if not (0.0 <= t < 1.0):
        raise ValueError("requires 0 <= t < 1")
    return (posterior_mean_x1 - x_t) / (1.0 - t)


@torch.inference_mode()
def empirical_velocity_t(
    *,
    core,
    x_t: torch.Tensor,
    reference: torch.Tensor,
    reference_norms: torch.Tensor,
    t: float,
    query_batch_size: int = 8,
    reference_chunk_size: int = 4096,
    reference_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Evaluate exact finite-S1 Flow-Matching empirical velocity on official t axis.

    For 0<t<1 the original empirical VE-style posterior is reused through the
    exact coordinate conversion

        y = x_t / t,  sigma = (1-t)/t.

    At t=0 the FM posterior over finite S1 is exactly uniform, so the velocity
    has the analytic form mean(S1)-x_0.  Supporting this endpoint is necessary
    for Experiment 2, whose empirical branch starts at the official FM origin.
    """
    t = float(t)
    if not (0.0 <= t < 1.0):
        raise ValueError("empirical_velocity_t requires 0 <= t < 1")

    if t == 0.0:
        if reference_mean is None:
            reference_mean = reference.mean(dim=0)
        mean = reference_mean.to(device=x_t.device, dtype=x_t.dtype).unsqueeze(0).expand_as(x_t)
        n = int(reference.shape[0])
        velocity = mean - x_t
        posterior = {
            "mean": mean,
            "ess": torch.full((len(x_t),), float(n), device=x_t.device, dtype=torch.float32),
            "top_probability": torch.full((len(x_t),), 1.0 / float(n), device=x_t.device, dtype=torch.float32),
        }
        return velocity, posterior

    sigma = sigma_from_t(t)
    y = x_t / t
    posterior = core.posterior_batched(
        y,
        reference,
        reference_norms,
        sigma,
        int(query_batch_size),
        int(reference_chunk_size),
    )
    velocity = empirical_velocity_from_mean(x_t, posterior["mean"], t)
    return velocity, posterior
