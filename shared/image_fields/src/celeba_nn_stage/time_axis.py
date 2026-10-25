from __future__ import annotations

import csv
import math
from pathlib import Path


def sigma_from_t(t: float) -> float:
    t = float(t)
    if t < 0 or t > 1:
        raise ValueError("t must lie in [0,1]")
    if t == 0:
        return math.inf
    return (1.0 - t) / t


def write_time_axis_reference(path: Path) -> None:
    # Includes the sigma values heavily used in the previous project plus
    # paper-relevant early t values. t is always the primary coordinate.
    t_values = sorted(set([
        0.0, 0.02, 0.05, 1/17, 0.10, 1/9, 0.15, 0.20, 1/5,
        0.25, 1/3, 0.40, 0.50, 2/3, 5/6, 10/11, 0.95, 0.98, 1.0,
    ]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_primary", "sigma_secondary=(1-t)/t"])
        for t in t_values:
            sigma = sigma_from_t(t)
            writer.writerow([f"{t:.10g}", "inf" if math.isinf(sigma) else f"{sigma:.10g}"])
