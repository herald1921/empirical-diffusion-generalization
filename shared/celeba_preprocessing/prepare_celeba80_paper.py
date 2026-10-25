#!/usr/bin/env python3
"""Prepare CelebA in the 80x80 grayscale representation used by the project.

The aligned 178x218 RGB JPEG is cropped with ``[29:-29, 9:-9]``, resized to
80x80 using PIL bicubic interpolation, and stored losslessly as the uint16 sum
of the three resized uint8 channels.  Runtime code decodes by division by 765.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from project_paths import CELEBA80_DATA_DIR
from paper_contract import DEFAULT_CONTRACT, load_contract, ordered_sources, validate_shard

FORMAT_NAME = "celeba80_gray_rgb_sum_shards_v1"
RAW_IMAGE_COUNT = 202_599
OFFICIAL_ZIP_MD5 = "00d2c5bc6d35e252742224ab0c1e8fcb"
EXPECTED_CLEAN_COUNT = 196_047
PAPER_HALF_SIZE = 98_023
PAPER_SUBSET_SIZES = (1, 10, 100, 1_000, 10_000, PAPER_HALF_SIZE)
IMAGE_SIZE = 80
DEDUP_SIZE = 20
DEDUP_THRESHOLD = 0.95
ImageSource = Union[zipfile.ZipInfo, Path]


def atomic_json(path: Path, value: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def file_hash(path: Path, algorithm: str, block_size: int = 8 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_image_entries(archive: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    entries = [
        info for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).suffix.lower() in {".jpg", ".jpeg"}
    ]
    if not entries:
        raise RuntimeError("No JPEG images were found in img_align_celeba.zip.")
    names = [Path(info.filename).name for info in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("The CelebA ZIP contains duplicate JPEG basenames.")
    return entries


def source_name(source: ImageSource) -> str:
    return Path(source.filename).name if isinstance(source, zipfile.ZipInfo) else source.name


def decode_one(archive: zipfile.ZipFile, source: ImageSource) -> np.ndarray:
    if isinstance(source, zipfile.ZipInfo):
        with archive.open(source, "r") as stream:
            context = Image.open(io.BytesIO(stream.read()))
        label = source.filename
    else:
        context, label = Image.open(source), str(source)
    with context as image:
        image = image.convert("RGB")
        if image.size != (178, 218):
            raise ValueError(f"Unexpected aligned CelebA size for {label}: {image.size}")
        image = image.crop((9, 29, 169, 189))
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resampling)
        rgb = np.asarray(image, dtype=np.uint8)
    return rgb.astype(np.uint16).sum(axis=2, dtype=np.uint16)


def extract_official_images(
    archive: zipfile.ZipFile,
    entries: Sequence[zipfile.ZipInfo],
    extract_dir: Path,
) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    for index, info in enumerate(entries, 1):
        destination = extract_dir / Path(info.filename).name
        if destination.exists():
            if not destination.is_file() or destination.stat().st_size != info.file_size:
                raise RuntimeError(f"Existing extracted file is incompatible: {destination}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                raise RuntimeError(f"Incomplete extraction marker exists: {temporary}")
            with archive.open(info) as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            os.replace(temporary, destination)
        if index % 5_000 == 0 or index == len(entries):
            print(f"extract: {index:,}/{len(entries):,}")


def author_os_listdir_sources(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Author-style image directory not found: {image_dir}")
    names = os.listdir(image_dir)  # Intentionally one unsorted call.
    if any(Path(name).suffix.lower() not in {".jpg", ".jpeg"} for name in names):
        raise RuntimeError("Author-style directory must contain JPEG files only.")
    sources = [image_dir / name for name in names]
    if any(not path.is_file() for path in sources):
        raise RuntimeError("Author-style directory contains a non-file entry.")
    return sources


def save_exact_order(output_dir: Path, names: Sequence[str]) -> str:
    payload = ("\n".join(names) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = output_dir / "raw_image_order.txt"
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"Image order differs from the existing contract: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return digest


def validate_output_contract(output_dir: Path, order_digest: str) -> None:
    manifest_path = output_dir / "manifest.json"
    has_shards = any(output_dir.glob("gray80_*.np*"))
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle).get("preprocessing", {}).get("order_sha256")
        if has_shards and previous != order_digest:
            raise RuntimeError("Existing shards use a different or unknown image order; use a new output directory.")
    elif has_shards:
        raise RuntimeError("Found shards without a manifest; use a new output directory.")


def save_shard(path: Path, images: np.ndarray, compression: str) -> None:
    if path.exists():
        if compression == "compressed":
            with np.load(path, allow_pickle=False) as archive:
                existing = archive["gray_rgb_sum"]
                shape, dtype = existing.shape, existing.dtype
        else:
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            shape, dtype = existing.shape, existing.dtype
            del existing
        if shape != images.shape or dtype != np.uint16:
            raise RuntimeError(f"Existing shard is incompatible: {path}")
        print(f"[resume] keeping existing shard {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        if compression == "compressed":
            np.savez_compressed(handle, gray_rgb_sum=images)
        else:
            np.save(handle, images, allow_pickle=False)
    os.replace(temporary, path)


def load_shard(path: Path, storage: str) -> np.ndarray:
    if storage == "npz_compressed":
        with np.load(path, allow_pickle=False) as archive:
            return np.ascontiguousarray(archive["gray_rgb_sum"])
    return np.ascontiguousarray(np.load(path, mmap_mode="r", allow_pickle=False))


def make_dedup_features(
    output_dir: Path,
    shards: Sequence[Dict[str, object]],
    num_images: int,
    storage: str,
) -> Path:
    feature_path = output_dir / "dedup_features_20x20_float32.npy"
    if feature_path.exists():
        existing = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if existing.shape == (num_images, DEDUP_SIZE**2) and existing.dtype == np.float32:
            return feature_path
        raise RuntimeError(f"Existing feature file is incompatible: {feature_path}")
    temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
    features = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(num_images, DEDUP_SIZE**2)
    )
    cursor = 0
    for shard in shards:
        images = load_shard(output_dir / str(shard["filename"]), storage)
        count = images.shape[0]
        pooled = images.astype(np.float32).reshape(count, 20, 4, 20, 4).mean((2, 4))
        flat = pooled.reshape(count, -1)
        flat -= flat.mean(1, keepdims=True)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms[norms == 0] = 0.001
        features[cursor : cursor + count] = flat / norms
        cursor += count
        print(f"features: {cursor:,}/{num_images:,}")
    features.flush()
    del features
    os.replace(temporary, feature_path)
    return feature_path


def resolve_dedup_device(requested: str) -> Optional[str]:
    if requested == "none":
        return None
    try:
        import torch
    except ImportError:
        if requested == "auto":
            return None
        raise RuntimeError("PyTorch is required for exact duplicate removal.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else None
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return requested


def exact_duplicate_mask(
    feature_path: Path,
    device_name: str,
    query_batch_size: int,
    reference_chunk_size: int,
    threshold: float,
) -> np.ndarray:
    import torch

    if device_name == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except (AttributeError, RuntimeError):
        pass
    source = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    n_images = source.shape[0]
    removed = np.zeros(n_images, dtype=np.bool_)
    device = torch.device(device_name)
    with torch.inference_mode():
        for q0 in range(0, n_images, query_batch_size):
            q1 = min(q0 + query_batch_size, n_images)
            query = torch.from_numpy(np.array(source[q0:q1], copy=True)).to(device)
            duplicate = torch.zeros(q1 - q0, dtype=torch.bool, device=device)
            row_ids = torch.arange(q0, q1, device=device).reshape(-1, 1)
            for r0 in range(0, q1, reference_chunk_size):
                r1 = min(r0 + reference_chunk_size, q1)
                reference = torch.from_numpy(np.array(source[r0:r1], copy=True)).to(device)
                scores = query @ reference.T
                col_ids = torch.arange(r0, r1, device=device).reshape(1, -1)
                duplicate |= scores.abs().masked_fill(col_ids >= row_ids, 0.0).amax(1) > threshold
            removed[q0:q1] = duplicate.cpu().numpy()
            print(f"dedup: {q1:,}/{n_images:,}; removed={removed[:q1].sum():,}")
    return removed


def save_partitions(output_dir: Path, removed: np.ndarray, expected_clean_count: Optional[int]) -> Dict[str, object]:
    keep = np.flatnonzero(~removed).astype(np.int64)
    removed_indices = np.flatnonzero(removed).astype(np.int64)
    if expected_clean_count is not None and keep.size != expected_clean_count:
        raise RuntimeError(f"Duplicate filtering produced {keep.size:,}; expected {expected_clean_count:,}.")
    half = PAPER_HALF_SIZE if keep.size == EXPECTED_CLEAN_COUNT else keep.size // 2
    np.save(output_dir / "keep_raw_indices.npy", keep, allow_pickle=False)
    np.save(output_dir / "removed_raw_indices.npy", removed_indices, allow_pickle=False)
    np.savez(
        output_dir / "paper_partitions.npz",
        group_a_raw_indices=keep[:half],
        group_b_raw_indices=keep[-half:],
        paper_subset_sizes=np.asarray(sorted(set(n for n in (*PAPER_SUBSET_SIZES, half) if n <= half)), dtype=np.int64),
    )
    return {
        "status": "complete", "threshold": DEDUP_THRESHOLD,
        "comparison": "absolute zero-mean cosine similarity at 20x20",
        "semantics": "remove row i if any original row j<i has abs(correlation)>threshold",
        "num_removed": int(removed_indices.size), "num_clean": int(keep.size),
        "paper_half_size": int(half), "keep_indices_file": "keep_raw_indices.npy",
        "removed_indices_file": "removed_raw_indices.npy", "partitions_file": "paper_partitions.npz",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeba-root", type=Path, required=True, help="Directory containing img_align_celeba.zip.")
    parser.add_argument("--output-dir", type=Path, default=CELEBA80_DATA_DIR)
    parser.add_argument("--ordering", choices=("paper-contract", "author-os-listdir", "zip", "lexicographic"), default="paper-contract")
    parser.add_argument("--paper-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--recompute-dedup", action="store_true", help="Recompute and require exact equality with released removed indices in paper-contract mode.")
    parser.add_argument("--author-image-dir", type=Path, default=None)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--compression", choices=("compressed", "none"), default="compressed")
    parser.add_argument("--dedup-device", choices=("auto", "cuda", "cpu", "none"), default="auto")
    parser.add_argument("--dedup-query-batch-size", type=int, default=1024)
    parser.add_argument("--dedup-reference-chunk-size", type=int, default=32768)
    parser.add_argument("--max-images", type=int, default=None, help="Smoke-test only.")
    parser.add_argument("--hash-source-zip", action="store_true")
    parser.add_argument("--skip-source-md5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive.")
    if args.dedup_query_batch_size < 1 or args.dedup_reference_chunk_size < 1:
        raise ValueError("Deduplication batch/chunk sizes must be positive.")
    source_zip = args.celeba_root.expanduser().resolve() / "img_align_celeba.zip"
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Use a new or empty output directory; existing experimental data and manifests are preserved.")
    contract, contract_names, contract_removed = (load_contract(args.paper_contract) if args.ordering == 'paper-contract' else (None, None, None))
    if contract is not None and args.shard_size != contract['shard_size']:
        raise ValueError('Paper-contract mode requires shard-size=4096 for per-shard pixel validation.')
    if not source_zip.is_file():
        raise FileNotFoundError(f"Official CelebA archive not found: {source_zip}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_md5 = None if args.skip_source_md5 else file_hash(source_zip, "md5")
    if source_md5 is not None and source_md5 != OFFICIAL_ZIP_MD5:
        raise RuntimeError(f"Unexpected ZIP MD5: {source_md5}; expected {OFFICIAL_ZIP_MD5}.")
    started = time.time()
    with zipfile.ZipFile(source_zip) as archive:
        entries = archive_image_entries(archive)
        if len(entries) != RAW_IMAGE_COUNT and args.max_images is None:
            raise RuntimeError(f"Expected {RAW_IMAGE_COUNT:,} images, found {len(entries):,}.")
        image_dir: Optional[Path] = None
        if args.ordering == "paper-contract":
            sources = ordered_sources(entries, contract_names)
        elif args.ordering == "author-os-listdir":
            image_dir = args.author_image_dir.expanduser().resolve() if args.author_image_dir else args.celeba_root.expanduser().resolve() / "img_align_celeba"
            extract_official_images(archive, entries, image_dir)
            sources: List[ImageSource] = author_os_listdir_sources(image_dir)
        elif args.ordering == "zip":
            sources = list(entries)
        else:
            sources = sorted(entries, key=lambda info: Path(info.filename).name)
        if args.max_images is not None:
            if args.max_images < 1:
                raise ValueError("max-images must be positive.")
            sources = sources[: args.max_images]
        order_digest = hashlib.sha256(("\n".join(map(source_name, sources)) + "\n").encode()).hexdigest()
        validate_output_contract(output_dir, order_digest)
        save_exact_order(output_dir, [source_name(source) for source in sources])
        shards: List[Dict[str, object]] = []
        suffix = ".npz" if args.compression == "compressed" else ".npy"
        for shard_id, start in enumerate(range(0, len(sources), args.shard_size)):
            end = min(start + args.shard_size, len(sources))
            filename = f"gray80_{shard_id:05d}{suffix}"
            path = output_dir / filename
            if path.exists():
                placeholder = np.empty((end - start, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
                save_shard(path, placeholder, args.compression)
            else:
                images = np.empty((end - start, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
                for local_index, source in enumerate(sources[start:end]):
                    images[local_index] = decode_one(archive, source)
                if contract is not None and args.max_images is None:
                    validate_shard(images, contract['shards'][shard_id])
                save_shard(path, images, args.compression)
            shards.append({
                "filename": filename, "start": start, "count": end - start,
                "first_image": source_name(sources[start]), "last_image": source_name(sources[end - 1]),
                "bytes": path.stat().st_size, "sha256": file_hash(path, "sha256"),
            })
            print(f"images: {end:,}/{len(sources):,}; wrote {filename}")
    storage = "npz_compressed" if args.compression == "compressed" else "npy"
    manifest: Dict[str, object] = {
        "format": FORMAT_NAME, "created_unix_time": time.time(), "source_zip": str(source_zip),
        "source_zip_bytes": source_zip.stat().st_size, "source_zip_md5": source_md5,
        "official_source_zip_md5": OFFICIAL_ZIP_MD5,
        "source_zip_sha256": file_hash(source_zip, "sha256") if args.hash_source_zip else None,
        "num_raw_images": len(sources), "official_full_count": RAW_IMAGE_COUNT,
        "image_shape": [1, IMAGE_SIZE, IMAGE_SIZE], "storage": storage,
        "pixel_encoding": "uint16 sum of resized uint8 RGB channels", "decode_divisor": 765.0,
        "preprocessing": {
            "input": "official aligned CelebA JPEG, 178x218 RGB",
            "crop": "NumPy [29:-29, 9:-9], equivalent PIL box (9,29,169,189)",
            "resize": "80x80 PIL bicubic", "grayscale": "RGB arithmetic mean after uint8 resize",
            "order": "unsorted os.listdir() called once" if args.ordering == "author-os-listdir" else args.ordering,
            "ordering_mode": args.ordering, "order_file": "raw_image_order.txt",
            "order_sha256": order_digest, "author_image_dir": str(image_dir) if image_dir else None,
            "historical_order_caveat": "paper-contract uses the released immutable order of this experiment; alternative order modes are exploratory.",
            "official_partition_file_used": False,
        },
        "paper": {"actual_half_size": None, "subset_sizes_actual": [], "contract_version": contract['version'] if contract else None, "scope": "smoke-only" if args.max_images else ("paper" if contract else "exploratory")},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pillow": getattr(Image, "__version__", "unknown")},
        "shards": shards,
        "deduplication": {"status": "pending", "threshold": DEDUP_THRESHOLD, "feature_file": "dedup_features_20x20_float32.npy"},
    }
    atomic_json(output_dir / "manifest.json", manifest)
    if contract is not None and not args.recompute_dedup:
        removed = np.zeros(len(sources), dtype=bool)
        removed[contract_removed[contract_removed < len(sources)]] = True
    else:
        feature_path = make_dedup_features(output_dir, shards, len(sources), storage)
        dedup_device = resolve_dedup_device(args.dedup_device)
        removed = None if dedup_device is None else exact_duplicate_mask(feature_path, dedup_device, args.dedup_query_batch_size, args.dedup_reference_chunk_size, DEDUP_THRESHOLD)
        if contract is not None:
            if removed is None or not np.array_equal(np.flatnonzero(removed), contract_removed[contract_removed < len(sources)]):
                raise RuntimeError('Recomputed duplicate mask differs from the immutable paper contract.')
    if removed is not None:
        expected = EXPECTED_CLEAN_COUNT if contract is not None and args.max_images is None else None
        manifest["deduplication"] = save_partitions(output_dir, removed, expected)
        half = manifest['deduplication']['paper_half_size']
        manifest['paper'].update(actual_half_size=half, subset_sizes_actual=sorted(set(n for n in (*PAPER_SUBSET_SIZES, half) if n <= half)))
        manifest['deduplication']['mask_source'] = 'released paper indices' if contract is not None and not args.recompute_dedup else 'recomputed'
        atomic_json(output_dir / "manifest.json", manifest)
    else:
        print("Exact duplicate removal deferred; the experiment runner will complete it.")
    print(f"Prepared data directory: {output_dir}")
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
