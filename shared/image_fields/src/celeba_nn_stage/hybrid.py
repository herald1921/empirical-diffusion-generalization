from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

VelocityFn = Callable[[torch.Tensor, float], torch.Tensor]


def _nn_velocity(model, x: torch.Tensor, t: float, autocast_bf16: bool) -> torch.Tensor:
    amp = bool(autocast_bf16 and x.device.type == "cuda")
    with torch.inference_mode(), torch.autocast(
        device_type=x.device.type,
        dtype=torch.bfloat16,
        enabled=amp,
    ):
        return model(x, float(t)).float()


def _midpoint_step(field: VelocityFn, x: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    """Second-order explicit midpoint step; never evaluates the field at t1."""
    dt = float(t1 - t0)
    k1 = field(x, float(t0))
    x_mid = x + 0.5 * dt * k1
    k2 = field(x_mid, float(t0 + 0.5 * dt))
    return x + dt * k2


@torch.inference_mode()
def integrate_experiment1_pair(
    model,
    x0: torch.Tensor,
    *,
    empirical_velocity: VelocityFn,
    switch_t: float = 0.8,
    steps: int = 100,
    capture_t: Iterable[float] = (),
    autocast_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[float, torch.Tensor], dict[float, torch.Tensor]]:
    """Experiment 1: compare full NN with NN->empirical after ``switch_t``.

    Both branches use exactly the same fixed source noise and exactly the same
    second-order midpoint grid.  They are identical on [0, switch_t].  At the
    switch the state is cloned; the baseline keeps the learned NN field while
    the intervention uses the exact empirical field for [switch_t, 1].

    Midpoint RK2 is used deliberately because the closed-form empirical FM
    velocity contains 1/(1-t), while midpoint integration reaches t=1 without
    ever evaluating the vector field at the singular endpoint itself.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    switch_t = float(switch_t)
    if not (0.0 < switch_t < 1.0):
        raise ValueError("switch_t must satisfy 0 < switch_t < 1")

    requested = sorted(set(float(v) for v in capture_t))
    if any(v < 0.0 or v > 1.0 for v in requested):
        raise ValueError("capture_t must lie in [0,1]")

    # Uniform official-t grid, with the switch and requested capture nodes
    # inserted exactly. This prevents a solver step from straddling the switch.
    base = [i / steps for i in range(steps + 1)]
    nodes = sorted(set(base + [switch_t] + requested))

    nn_field: VelocityFn = lambda x, t: _nn_velocity(model, x, t, autocast_bf16)

    x_nn = x0.float().clone()
    x_hybrid = x0.float().clone()
    cap_nn: dict[float, torch.Tensor] = {}
    cap_hybrid: dict[float, torch.Tensor] = {}
    request_set = set(requested)
    if 0.0 in request_set:
        cap_nn[0.0] = x_nn.detach().cpu()
        cap_hybrid[0.0] = x_hybrid.detach().cpu()

    for t0, t1 in zip(nodes[:-1], nodes[1:]):
        # Full-NN control always uses the learned field.
        x_nn = _midpoint_step(nn_field, x_nn, t0, t1)

        # Intervention is identical before switch; exact empirical afterwards.
        if t0 < switch_t - 1e-12:
            x_hybrid = _midpoint_step(nn_field, x_hybrid, t0, t1)
        else:
            x_hybrid = _midpoint_step(empirical_velocity, x_hybrid, t0, t1)

        if t1 in request_set:
            cap_nn[t1] = x_nn.detach().cpu()
            cap_hybrid[t1] = x_hybrid.detach().cpu()

    return x_nn, x_hybrid, cap_nn, cap_hybrid


@torch.inference_mode()
def integrate_field_midpoint(
    x0: torch.Tensor,
    *,
    field: VelocityFn,
    start_t: float = 0.0,
    end_t: float = 1.0,
    steps: int = 100,
    capture_t: Iterable[float] = (),
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Integrate an arbitrary field on a global uniform-t grid with exact capture nodes.

    ``steps`` denotes the number of intervals on [0,1].  The sub-interval
    [start_t,end_t] uses all matching global grid nodes plus requested captures.
    Midpoint RK2 never evaluates the field at ``end_t`` itself, which is useful
    for the exact empirical FM field at t=1.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    start_t = float(start_t)
    end_t = float(end_t)
    if not (0.0 <= start_t < end_t <= 1.0):
        raise ValueError("require 0 <= start_t < end_t <= 1")
    requested = sorted(set(float(v) for v in capture_t if start_t <= float(v) <= end_t))
    base = [i / steps for i in range(steps + 1) if start_t <= i / steps <= end_t]
    nodes = sorted(set(base + [start_t, end_t] + requested))
    x = x0.float().clone()
    caps: dict[float, torch.Tensor] = {}
    request_set = set(requested)
    if start_t in request_set:
        caps[start_t] = x.detach().cpu()
    for t0, t1 in zip(nodes[:-1], nodes[1:]):
        x = _midpoint_step(field, x, t0, t1)
        if t1 in request_set:
            caps[t1] = x.detach().cpu()
    return x, caps


@torch.inference_mode()
def integrate_nn_suffix(
    model,
    x_switch: torch.Tensor,
    *,
    switch_t: float = 0.8,
    steps: int = 100,
    capture_t: Iterable[float] = (),
    autocast_bf16: bool = True,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Continue a precomputed state from ``switch_t`` to 1 using the learned NN field."""
    nn_field: VelocityFn = lambda x, t: _nn_velocity(model, x, t, autocast_bf16)
    return integrate_field_midpoint(
        x_switch,
        field=nn_field,
        start_t=float(switch_t),
        end_t=1.0,
        steps=steps,
        capture_t=capture_t,
    )
