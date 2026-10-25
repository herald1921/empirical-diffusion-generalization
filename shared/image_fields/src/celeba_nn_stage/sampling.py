from __future__ import annotations

from collections.abc import Iterable
import torch


def make_fixed_noise(num_samples: int, dimension: int = 6400, seed: int = 20260810) -> torch.Tensor:
    """Create fixed CPU Gaussian source samples shared by every checkpoint."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(num_samples, dimension, generator=g, dtype=torch.float32)


@torch.inference_mode()
def integrate_heun(
    model,
    x0: torch.Tensor,
    *,
    steps: int = 80,
    capture_t: Iterable[float] = (),
    autocast_bf16: bool = True,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Integrate dx/dt = u_theta(x,t) from t=0 to t=1 with Heun.

    The official axis is t. Capture times are snapped to the nearest solver grid
    point and returned under their requested float keys.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = x0.device
    x = x0.float().clone()
    grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)

    requested = [float(v) for v in capture_t]
    if any(v < 0.0 or v > 1.0 for v in requested):
        raise ValueError("capture_t must lie in [0,1]")
    snap = {int(round(v * steps)): v for v in requested}
    captures: dict[float, torch.Tensor] = {}
    if 0 in snap:
        captures[snap[0]] = x.detach().cpu()

    amp = bool(autocast_bf16 and device.type == "cuda")
    for i in range(steps):
        t0 = grid[i]
        t1 = grid[i + 1]
        dt = t1 - t0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            v0 = model(x, t0)
        x_euler = x + dt * v0.float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            v1 = model(x_euler, t1)
        x = x + 0.5 * dt * (v0.float() + v1.float())
        j = i + 1
        if j in snap:
            captures[snap[j]] = x.detach().cpu()
    return x, captures
