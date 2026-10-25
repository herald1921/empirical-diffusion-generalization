#!/usr/bin/env python3
"""Fast structural and checksum verification for the prepared cloud data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from project_paths import CELEBA80_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CELEBA80_DATA_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_shape_dtype(path: Path, storage: str) -> tuple[tuple[int, ...], np.dtype]:
    if storage == "npz_compressed":
        with np.load(path, allow_pickle=False) as archive:
            values = archive["gray_rgb_sum"]
            return values.shape, values.dtype
    if storage == "npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        shape, dtype = values.shape, values.dtype
        del values
        return shape, dtype
    raise RuntimeError(f"Unsupported storage mode in manifest: {storage}")


def main() -> None:
    root = parse_args().data_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected_count = int(manifest.get("official_full_count", 202_599))
    if int(manifest["num_raw_images"]) != expected_count:
        raise RuntimeError(f"Prepared image count is not {expected_count:,}.")
    if manifest["image_shape"] != [1, 80, 80]:
        raise RuntimeError(f"Unexpected image shape: {manifest['image_shape']}")

    order_payload = (root / "raw_image_order.txt").read_bytes()
    order_count = len(order_payload.splitlines())
    if order_count != expected_count:
        raise RuntimeError(f"Image-order file has {order_count} rows; expected {expected_count}.")
    order_sha = hashlib.sha256(order_payload).hexdigest()
    expected_order_sha = manifest["preprocessing"]["order_sha256"]
    if order_sha != expected_order_sha:
        raise RuntimeError("Image-order SHA256 mismatch.")

    storage = str(manifest["storage"])
    total = 0
    for index, info in enumerate(manifest["shards"], start=1):
        path = root / info["filename"]
        if sha256_file(path) != info["sha256"]:
            raise RuntimeError(f"Shard checksum mismatch: {path.name}")
        shape, dtype = shard_shape_dtype(path, storage)
        expected = (int(info["count"]), 80, 80)
        if shape != expected or dtype != np.uint16:
            raise RuntimeError(f"Bad shard {path.name}: {shape}, {dtype}")
        total += int(info["count"])
        if index % 10 == 0:
            print(f"verified shards: {index}/{len(manifest['shards'])}")

    features = np.load(
        root / "dedup_features_20x20_float32.npy", mmap_mode="r", allow_pickle=False
    )
    if features.shape != (expected_count, 400) or features.dtype != np.float32:
        raise RuntimeError(f"Bad duplicate features: {features.shape}, {features.dtype}")
    if total != expected_count:
        raise RuntimeError(f"Shard counts sum to {total}; expected {expected_count}.")

    dedup = manifest["deduplication"]
    if dedup.get("status") == "complete":
        partition_path = root / str(dedup["partitions_file"])
        with np.load(partition_path, allow_pickle=False) as archive:
            group_a = archive["group_a_raw_indices"]
            group_b = archive["group_b_raw_indices"]
        if group_a.ndim != 1 or group_b.ndim != 1 or group_a.size != group_b.size:
            raise RuntimeError("Prepared paper partitions are malformed or unbalanced.")

    print("Prepared CelebA data verification passed.")
    print(f"images={total:,}, shards={len(manifest['shards'])}, shape=1x80x80, storage={storage}")
    print(f"ordering_sha256={order_sha}")
    print(f"deduplication_status={dedup['status']}")


if __name__ == "__main__":
    main()
