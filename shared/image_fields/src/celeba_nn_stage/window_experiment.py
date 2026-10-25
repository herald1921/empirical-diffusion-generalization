"""Matched-grid short-window interventions on the official Flow Matching t axis.

All branches must reuse one grid built with every intervention boundary and
capture time. Source samples are supplied by the caller; this module never
perturbs them, clips states, or creates noise.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

VelocityFn = Callable[[torch.Tensor, float], torch.Tensor]
EmpiricalFn = Callable[[torch.Tensor, float], tuple[torch.Tensor, Mapping]]


def _time(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("times must be finite and lie in [0,1]")
    return round(value, 12) + 0.0  # Normalize negative zero for stable keys.


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _time(self.start))
        object.__setattr__(self, "end", _time(self.end))
        if self.start >= self.end:
            raise ValueError("window requires start < end at 12-decimal precision")

    @property
    def key(self) -> str:
        def label(t: float) -> str:
            return f"{t:.12f}".rstrip("0").rstrip(".").replace(".", "p")

        return f"w_{label(self.start)}_{label(self.end)}"


def build_grid(
    steps: int,
    windows: Iterable[Window] = (),
    capture_t: Iterable[float] = (),
) -> tuple[float, ...]:
    """Union uniform nodes, all window boundaries and captures, rounded to 12 dp."""
    try:
        count = operator.index(steps)
    except TypeError as exc:
        raise ValueError("steps must be a positive integer") from exc
    if isinstance(steps, bool) or count <= 0:
        raise ValueError("steps must be a positive integer")
    nodes = {_time(i / count) for i in range(count + 1)}
    for window in windows:
        nodes.update((window.start, window.end))
    nodes.update(_time(t) for t in capture_t)
    return tuple(sorted(nodes))


def _grid(grid: Iterable[float]) -> tuple[float, ...]:
    nodes = tuple(_time(t) for t in grid)
    if not nodes or any(a >= b for a, b in zip(nodes, nodes[1:])):
        raise ValueError("grid must be nonempty and strictly increasing at 12-decimal precision")
    return nodes


def _captures(capture_t: Iterable[float], nodes: tuple[float, ...]) -> set[float]:
    requested = {_time(t) for t in capture_t}
    missing = requested.difference(nodes)
    if missing:
        raise ValueError(f"capture times must be explicit shared grid nodes: {sorted(missing)}")
    return requested


def _finite(x: torch.Tensor, description: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"non-finite {description}")


@torch.inference_mode()
def integrate_grid(
    field: VelocityFn,
    x0: torch.Tensor,
    grid: Iterable[float],
    capture_t: Iterable[float] = (),
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Explicit midpoint RK2 on supplied nodes; never evaluate the field at t=1.

    ``x0`` is the state at the first supplied node. Subgrids are allowed for
    baseline prefixes and suffixes. Captures are independent CPU snapshots;
    the final state retains the input device and floating dtype.
    """
    nodes = _grid(grid)
    requested = _captures(capture_t, nodes)
    if not x0.is_floating_point():
        raise ValueError("x0 must be a floating-point tensor")
    _finite(x0, "initial state")
    x = x0.detach().clone()
    caps = {}
    if nodes[0] in requested:
        caps[nodes[0]] = x.detach().cpu().clone()
    for t0, t1 in zip(nodes, nodes[1:]):
        dt = t1 - t0
        k1 = field(x, t0)
        midpoint = x + (0.5 * dt) * k1
        k2 = field(midpoint, t0 + 0.5 * dt)
        x = x + dt * k2
        if t1 in requested:
            _finite(x, f"captured state at t={t1}")
            caps[t1] = x.detach().cpu().clone()
    _finite(x, "final state")
    return x, caps


@torch.inference_mode()
def run_window(
    nn_field: VelocityFn,
    emp_field: VelocityFn,
    x_start: torch.Tensor,
    grid: Iterable[float],
    window: Window,
    direction: str,
    capture_t: Iterable[float] = (),
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Continue a common baseline state, swapping the field only on [start,end).

    ``nn_emp_nn`` starts from the NN baseline at ``window.start``; ``emp_nn_emp``
    starts from the empirical baseline. Every boundary and capture must already
    belong to the global 0..1 grid. Captures before start are validated but left
    to the caller's common baseline prefix.
    """
    if direction == "nn_emp_nn":
        baseline, alternate = nn_field, emp_field
    elif direction == "emp_nn_emp":
        baseline, alternate = emp_field, nn_field
    else:
        raise ValueError("direction must be 'nn_emp_nn' or 'emp_nn_emp'")
    nodes = _grid(grid)
    if nodes[0] != 0.0 or nodes[-1] != 1.0:
        raise ValueError("run_window requires the full shared grid from 0 to 1")
    if window.start not in nodes or window.end not in nodes:
        raise ValueError("window boundaries must be explicit shared grid nodes")
    requested = _captures(capture_t, nodes)
    suffix = nodes[nodes.index(window.start) :]

    def field(x: torch.Tensor, t: float) -> torch.Tensor:
        active = alternate if window.start <= t < window.end else baseline
        return active(x, t)

    return integrate_grid(field, x_start, suffix, (t for t in requested if t >= window.start))


@torch.inference_mode()
def compare_clean(
    nn_field: VelocityFn,
    emp_eval: EmpiricalFn,
    x: torch.Tensor,
    t: float,
) -> dict[str, torch.Tensor]:
    """Compare both denoisers and velocities queried at the identical raw state.

    NN clean estimate is x+(1-t)v_nn; the empirical estimate is the exact
    posterior mean. ``difference`` means NN minus empirical, and ``cosine``
    is velocity cosine similarity (zero when a velocity has zero norm).
    All results are detached CPU tensors; scalar diagnostics are per sample.
    """
    t = float(t)
    if not math.isfinite(t) or not 0.0 <= t < 1.0:
        raise ValueError("clean comparison requires finite 0 <= t < 1")
    if x.ndim < 2 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor with batch and feature dimensions")
    _finite(x, "comparison state")
    v_nn = nn_field(x, t)
    v_emp, posterior = emp_eval(x, t)
    clean_emp = posterior["mean"]
    if v_nn.shape != x.shape or v_emp.shape != x.shape or clean_emp.shape != x.shape:
        raise ValueError("both velocities and posterior mean must match x.shape")
    clean_nn = x + (1.0 - t) * v_nn
    difference = clean_nn - clean_emp
    nn_flat = v_nn.reshape(len(x), -1)
    emp_flat = v_emp.reshape(len(x), -1)
    result = {
        "state": x,
        "clean_nn": clean_nn,
        "clean_emp": clean_emp,
        "difference": difference,
        "denoiser_rmse": difference.reshape(len(x), -1).square().mean(1).sqrt(),
        "velocity_rmse": (nn_flat - emp_flat).square().mean(1).sqrt(),
        "cosine": F.cosine_similarity(nn_flat, emp_flat, dim=1),
        "ess": posterior["ess"],
        "top_probability": posterior["top_probability"],
    }
    for name, value in result.items():
        _finite(value, name)
    return {name: value.detach().cpu().clone() for name, value in result.items()}
