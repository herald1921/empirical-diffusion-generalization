from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def _milestone_candidates(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint_step_*.pt"):
        m = _STEP_RE.search(path.name)
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out, reverse=True)


def find_best_resume(checkpoint_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Return the valid checkpoint with the greatest internal step.

    checkpoint_latest.pt is loaded to inspect its internal step. Permanent
    milestone files use their filename step. Corrupt candidates are skipped.
    """
    checkpoint_dir = Path(checkpoint_dir)
    choices: list[tuple[int, Path, dict[str, Any] | None]] = []
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.is_file():
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
            choices.append((int(payload["step"]), latest, payload))
        except Exception as exc:
            print(f"[resume-warning] cannot read {latest}: {exc}", flush=True)
    for step, path in _milestone_candidates(checkpoint_dir):
        choices.append((step, path, None))
    choices.sort(key=lambda x: x[0], reverse=True)

    for expected_step, path, payload in choices:
        try:
            if payload is None:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            actual = int(payload["step"])
            if actual != expected_step and path.name != "checkpoint_latest.pt":
                raise RuntimeError(f"filename says {expected_step}, payload says {actual}")
            return path, payload
        except Exception as exc:
            print(f"[resume-warning] skipping invalid {path}: {exc}", flush=True)
    return None, None
