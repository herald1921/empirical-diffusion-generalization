from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_BASE_REPO = Path("/root/shared-nvme/experiment-mnist/celeba80_cloud_bundle")
DEFAULT_DATA_DIR = Path("/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order")


@dataclass(frozen=True)
class DatasetInfo:
    group: str
    count: int
    dimension: int
    image_shape: tuple[int, int]
    data_dir: str
    base_repo: str
    decode_divisor: float


def _load_core(base_repo: Path):
    base_repo = Path(base_repo).expanduser().resolve()
    core_file = base_repo / "celeba80_theory_core.py"
    if not core_file.is_file():
        raise FileNotFoundError(
            f"Cannot find original project core: {core_file}\n"
            "Set --base-repo to the existing celeba80_cloud_bundle checkout."
        )
    text = str(base_repo)
    if text not in sys.path:
        sys.path.insert(0, text)
    return importlib.import_module("celeba80_theory_core")


def load_full_s1(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    base_repo: Path = DEFAULT_BASE_REPO,
    device: torch.device,
) -> tuple[torch.Tensor, DatasetInfo]:
    """Load the original project's complete S1 split exactly as stored.

    The split size is intentionally NOT hard-coded.  ``paper_partitions.npz``
    is the source of truth, so the training code follows the exact S1 used by
    the installed ``celeba80_cloud_bundle`` checkout.  This preserves image
    ordering, crop/resize representation, uint16 RGB-sum decoding and S1/S2
    partition semantics without silently changing the dataset.
    """
    core = _load_core(base_repo)
    data_dir = Path(data_dir).expanduser().resolve()
    store = core.CelebAStore(data_dir)

    # Load through the original data contract on CPU first, then move only S1
    # to CUDA. This avoids keeping S2 resident on the 24-GB GPU.
    s1_cpu, s2_cpu = store.load_groups(torch.device("cpu"))
    del s2_cpu
    if tuple(s1_cpu.shape[1:]) != (80 * 80,):
        raise RuntimeError(f"Unexpected S1 shape: {tuple(s1_cpu.shape)}")

    s1 = s1_cpu.to(device=device, dtype=torch.float32, non_blocking=False)
    del s1_cpu
    info = DatasetInfo(
        group="S1",
        count=len(s1),
        dimension=int(s1.shape[1]),
        image_shape=(80, 80),
        data_dir=str(data_dir),
        base_repo=str(Path(base_repo).expanduser().resolve()),
        decode_divisor=float(store.decode_divisor),
    )
    return s1, info


def load_full_groups_cpu(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    base_repo: Path = DEFAULT_BASE_REPO,
) -> tuple[torch.Tensor, torch.Tensor, DatasetInfo, DatasetInfo]:
    """Load unchanged S1 and S2 on CPU from the original CelebAStore contract."""
    core = _load_core(base_repo)
    data_dir = Path(data_dir).expanduser().resolve()
    store = core.CelebAStore(data_dir)
    s1, s2 = store.load_groups(torch.device("cpu"))
    if tuple(s1.shape[1:]) != (80 * 80,) or tuple(s2.shape[1:]) != (80 * 80,):
        raise RuntimeError(f"Unexpected group shapes: S1={tuple(s1.shape)}, S2={tuple(s2.shape)}")
    base = str(Path(base_repo).expanduser().resolve())
    common = dict(
        dimension=int(s1.shape[1]),
        image_shape=(80, 80),
        data_dir=str(data_dir),
        base_repo=base,
        decode_divisor=float(store.decode_divisor),
    )
    i1 = DatasetInfo(group="S1", count=len(s1), **common)
    i2 = DatasetInfo(group="S2", count=len(s2), **common)
    return s1.float(), s2.float(), i1, i2
