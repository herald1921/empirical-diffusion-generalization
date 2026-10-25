#!/usr/bin/env python3
"""Memory-safe Figure-2 gallery selection using centered cosine similarity.

This is the single implementation used by generation and plotting scripts. It
preserves the result-field API of the original repository, but removes patch-
installer code and keeps every expensive comparison block-vectorized.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Tuple

import numpy as np

NORM_FLOOR = 0.001
DEFAULT_MAX_SIMILARITY_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class BestCrossPair:
    index_a: int
    index_b: int
    similarity: float


@dataclass(frozen=True)
class BestPairedSeed:
    index: int
    similarity: float


@dataclass(frozen=True)
class MedianPairedSeed:
    index: int
    similarity: float
    median_similarity: float


@dataclass(frozen=True)
class GallerySelection:
    nearest_train_a: np.ndarray
    generated_a: np.ndarray
    generated_b: np.ndarray
    nearest_train_b: np.ndarray
    index_a: int
    index_b: int
    similarity: float
    nearest_train_index_a: int
    nearest_train_index_b: int
    source: str

    @property
    def images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.nearest_train_a,
            self.generated_a,
            self.generated_b,
            self.nearest_train_b,
        )


def _to_numpy(value: object, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()  # type: ignore[union-attr]
    if hasattr(value, "cpu"):
        value = value.cpu()  # type: ignore[union-attr]
    if hasattr(value, "numpy"):
        value = value.numpy()  # type: ignore[union-attr]
    return np.asarray(value, dtype=dtype)


def _rows(array: object) -> np.ndarray:
    values = _to_numpy(array, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError(f"Expected [samples, ...], got shape {values.shape}.")
    if values.shape[0] < 1:
        raise ValueError("At least one sample is required in each generated set.")
    return np.ascontiguousarray(values.reshape(values.shape[0], -1))


def centered_normalize_rows(array: object) -> np.ndarray:
    values = _rows(array)
    centered = values - values.mean(axis=1, keepdims=True, dtype=np.float32)
    norms = np.sqrt(
        np.sum(centered * centered, axis=1, keepdims=True, dtype=np.float32)
    ).astype(np.float32, copy=False)
    return centered / np.maximum(norms, np.float32(NORM_FLOOR))


def centered_cosine_pair(image_a: object, image_b: object) -> float:
    rows_a = _rows(np.asarray(_to_numpy(image_a))[None, ...])
    rows_b = _rows(np.asarray(_to_numpy(image_b))[None, ...])
    if rows_a.shape[1] != rows_b.shape[1]:
        raise ValueError(f"Image dimensions differ: {rows_a.shape[1]} vs {rows_b.shape[1]}.")
    normalized_a = centered_normalize_rows(rows_a)
    normalized_b = centered_normalize_rows(rows_b)
    return float(np.clip(np.sum(normalized_a * normalized_b, dtype=np.float32), -1.0, 1.0))


def paired_centered_cosine_values(
    generated_a: object,
    generated_b: object,
    *,
    block_size: int = 4096,
) -> np.ndarray:
    rows_a = _rows(generated_a)
    rows_b = _rows(generated_b)
    if rows_a.shape != rows_b.shape:
        raise ValueError(f"Paired arrays differ: {rows_a.shape} vs {rows_b.shape}.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    result = np.empty(rows_a.shape[0], dtype=np.float32)
    for start in range(0, rows_a.shape[0], int(block_size)):
        end = min(start + int(block_size), rows_a.shape[0])
        normalized_a = centered_normalize_rows(rows_a[start:end])
        normalized_b = centered_normalize_rows(rows_b[start:end])
        result[start:end] = np.clip(
            np.einsum("ij,ij->i", normalized_a, normalized_b, optimize=True), -1.0, 1.0
        )
    return result


def best_paired_seed_centered_cosine(
    generated_a: object,
    generated_b: object,
    *,
    block_size: int = 4096,
) -> BestPairedSeed:
    values = paired_centered_cosine_values(generated_a, generated_b, block_size=block_size)
    index = int(np.argmax(values))
    return BestPairedSeed(index=index, similarity=float(values[index]))


def median_paired_seed_centered_cosine(
    generated_a: object,
    generated_b: object,
    *,
    block_size: int = 4096,
) -> MedianPairedSeed:
    values = paired_centered_cosine_values(generated_a, generated_b, block_size=block_size)
    median = float(np.median(values))
    index = int(np.argmin(np.abs(values - median)))
    return MedianPairedSeed(index=index, similarity=float(values[index]), median_similarity=median)


def _block_shape(n_a: int, n_b: int, max_elements: int) -> Tuple[int, int]:
    if max_elements < 1:
        raise ValueError("max_similarity_elements must be positive.")
    block_a = min(n_a, max(1, int(math.sqrt(max_elements))))
    block_b = min(n_b, max(1, max_elements // block_a))
    return block_a, block_b


def best_cross_set_centered_cosine(
    generated_a: object,
    generated_b: object,
    *,
    max_similarity_elements: int = DEFAULT_MAX_SIMILARITY_ELEMENTS,
) -> BestCrossPair:
    rows_a = _rows(generated_a)
    rows_b = _rows(generated_b)
    if rows_a.shape[1] != rows_b.shape[1]:
        raise ValueError(f"Generated dimensions differ: {rows_a.shape[1]} vs {rows_b.shape[1]}.")
    block_a, block_b = _block_shape(rows_a.shape[0], rows_b.shape[0], int(max_similarity_elements))
    best_value = -math.inf
    best_i = best_j = -1
    for a0 in range(0, rows_a.shape[0], block_a):
        a1 = min(a0 + block_a, rows_a.shape[0])
        normalized_a = centered_normalize_rows(rows_a[a0:a1])
        for b0 in range(0, rows_b.shape[0], block_b):
            b1 = min(b0 + block_b, rows_b.shape[0])
            normalized_b = centered_normalize_rows(rows_b[b0:b1])
            similarity_matrix = np.clip(normalized_a @ normalized_b.T, -1.0, 1.0)
            local_flat = int(np.argmax(similarity_matrix))
            local_i, local_j = np.unravel_index(local_flat, similarity_matrix.shape)
            value = float(similarity_matrix[local_i, local_j])
            global_i, global_j = a0 + int(local_i), b0 + int(local_j)
            if value > best_value or (
                value == best_value and (global_i, global_j) < (best_i, best_j)
            ):
                best_value, best_i, best_j = value, global_i, global_j
    return BestCrossPair(best_i, best_j, best_value)


def _take_reference(reference: object, index: int) -> np.ndarray:
    return np.array(_to_numpy(reference[index], dtype=np.float32).reshape(-1), copy=True)  # type: ignore[index]


def build_best_cross_payload(
    generated_a: object,
    generated_b: object,
    nearest_index_a: object,
    nearest_index_b: object,
    reference_a: object,
    reference_b: object,
    *,
    prefix: str = "",
    max_similarity_elements: int = DEFAULT_MAX_SIMILARITY_ELEMENTS,
) -> MutableMapping[str, np.ndarray]:
    rows_a, rows_b = _rows(generated_a), _rows(generated_b)
    pair = best_cross_set_centered_cosine(
        rows_a, rows_b, max_similarity_elements=max_similarity_elements
    )
    nearest_a = _to_numpy(nearest_index_a, dtype=np.int64).reshape(-1)
    nearest_b = _to_numpy(nearest_index_b, dtype=np.int64).reshape(-1)
    train_index_a, train_index_b = int(nearest_a[pair.index_a]), int(nearest_b[pair.index_b])
    stem = f"{prefix}_" if prefix else ""
    return {
        f"{stem}best_cross_index_a": np.asarray(pair.index_a, dtype=np.int64),
        f"{stem}best_cross_index_b": np.asarray(pair.index_b, dtype=np.int64),
        f"{stem}best_cross_similarity": np.asarray(pair.similarity, dtype=np.float32),
        f"{stem}best_cross_generated_a": np.array(rows_a[pair.index_a], dtype=np.float32, copy=True),
        f"{stem}best_cross_generated_b": np.array(rows_b[pair.index_b], dtype=np.float32, copy=True),
        f"{stem}best_cross_nearest_train_a": _take_reference(reference_a, train_index_a),
        f"{stem}best_cross_nearest_train_b": _take_reference(reference_b, train_index_b),
        f"{stem}best_cross_nearest_train_index_a": np.asarray(train_index_a, dtype=np.int64),
        f"{stem}best_cross_nearest_train_index_b": np.asarray(train_index_b, dtype=np.int64),
        f"{stem}best_cross_search_count_a": np.asarray(rows_a.shape[0], dtype=np.int64),
        f"{stem}best_cross_search_count_b": np.asarray(rows_b.shape[0], dtype=np.int64),
    }


def _build_named_paired_payload(
    generated_a: object,
    generated_b: object,
    nearest_index_a: object,
    nearest_index_b: object,
    reference_a: object,
    reference_b: object,
    *,
    prefix: str,
    block_size: int,
    selection_kind: str,
) -> MutableMapping[str, np.ndarray]:
    rows_a, rows_b = _rows(generated_a), _rows(generated_b)
    if selection_kind == "best":
        selected = best_paired_seed_centered_cosine(rows_a, rows_b, block_size=block_size)
        stem_kind, target = "best_paired", None
    elif selection_kind == "median":
        median_selected = median_paired_seed_centered_cosine(rows_a, rows_b, block_size=block_size)
        selected = median_selected
        stem_kind, target = "median_paired", median_selected.median_similarity
    else:
        raise ValueError(f"Unknown selection_kind: {selection_kind}")
    nearest_a = _to_numpy(nearest_index_a, dtype=np.int64).reshape(-1)
    nearest_b = _to_numpy(nearest_index_b, dtype=np.int64).reshape(-1)
    if nearest_a.size != rows_a.shape[0] or nearest_b.size != rows_b.shape[0]:
        raise ValueError("Nearest-index arrays must align with generated samples.")
    train_index_a = int(nearest_a[selected.index])
    train_index_b = int(nearest_b[selected.index])
    stem = f"{prefix}_" if prefix else ""
    payload: MutableMapping[str, np.ndarray] = {
        f"{stem}{stem_kind}_index": np.asarray(selected.index, dtype=np.int64),
        f"{stem}{stem_kind}_index_a": np.asarray(selected.index, dtype=np.int64),
        f"{stem}{stem_kind}_index_b": np.asarray(selected.index, dtype=np.int64),
        f"{stem}{stem_kind}_similarity": np.asarray(selected.similarity, dtype=np.float32),
        f"{stem}{stem_kind}_generated_a": np.array(rows_a[selected.index], dtype=np.float32, copy=True),
        f"{stem}{stem_kind}_generated_b": np.array(rows_b[selected.index], dtype=np.float32, copy=True),
        f"{stem}{stem_kind}_nearest_train_a": _take_reference(reference_a, train_index_a),
        f"{stem}{stem_kind}_nearest_train_b": _take_reference(reference_b, train_index_b),
        f"{stem}{stem_kind}_nearest_train_index_a": np.asarray(train_index_a, dtype=np.int64),
        f"{stem}{stem_kind}_nearest_train_index_b": np.asarray(train_index_b, dtype=np.int64),
        f"{stem}{stem_kind}_search_count": np.asarray(rows_a.shape[0], dtype=np.int64),
    }
    if target is not None:
        payload[f"{stem}{stem_kind}_target_similarity"] = np.asarray(target, dtype=np.float32)
    return payload


def build_best_paired_payload(
    generated_a: object,
    generated_b: object,
    nearest_index_a: object,
    nearest_index_b: object,
    reference_a: object,
    reference_b: object,
    *,
    prefix: str = "",
    block_size: int = 4096,
) -> MutableMapping[str, np.ndarray]:
    return _build_named_paired_payload(
        generated_a, generated_b, nearest_index_a, nearest_index_b,
        reference_a, reference_b, prefix=prefix, block_size=block_size,
        selection_kind="best",
    )


def build_median_paired_payload(
    generated_a: object,
    generated_b: object,
    nearest_index_a: object,
    nearest_index_b: object,
    reference_a: object,
    reference_b: object,
    *,
    prefix: str = "",
    block_size: int = 4096,
) -> MutableMapping[str, np.ndarray]:
    return _build_named_paired_payload(
        generated_a, generated_b, nearest_index_a, nearest_index_b,
        reference_a, reference_b, prefix=prefix, block_size=block_size,
        selection_kind="median",
    )


def _field_names(prefix: str, kind: str, suffixes: Tuple[str, ...]) -> Tuple[str, ...]:
    stem = f"{prefix}_" if prefix else ""
    return tuple(f"{stem}{kind}_{suffix}" for suffix in suffixes)


def best_paired_field_names(prefix: str = "") -> Tuple[str, ...]:
    return _field_names(prefix, "best_paired", (
        "index", "index_a", "index_b", "similarity", "generated_a", "generated_b",
        "nearest_train_a", "nearest_train_b", "nearest_train_index_a",
        "nearest_train_index_b", "search_count",
    ))


def median_paired_field_names(prefix: str = "") -> Tuple[str, ...]:
    return _field_names(prefix, "median_paired", (
        "index", "index_a", "index_b", "similarity", "generated_a", "generated_b",
        "nearest_train_a", "nearest_train_b", "nearest_train_index_a",
        "nearest_train_index_b", "search_count", "target_similarity",
    ))


def best_cross_field_names(prefix: str = "") -> Tuple[str, ...]:
    return _field_names(prefix, "best_cross", (
        "index_a", "index_b", "similarity", "generated_a", "generated_b",
        "nearest_train_a", "nearest_train_b", "nearest_train_index_a",
        "nearest_train_index_b", "search_count_a", "search_count_b",
    ))


def best_paired_payload_available(mapping: Mapping[str, np.ndarray], prefix: str = "") -> bool:
    return all(key in mapping for key in best_paired_field_names(prefix)[:10])


def median_paired_payload_available(mapping: Mapping[str, np.ndarray], prefix: str = "") -> bool:
    return all(key in mapping for key in median_paired_field_names(prefix)[:10])


def best_cross_payload_available(mapping: Mapping[str, np.ndarray], prefix: str = "") -> bool:
    return all(key in mapping for key in best_cross_field_names(prefix)[:9])


def _scalar(mapping: Mapping[str, np.ndarray], key: str, cast):
    return cast(np.asarray(mapping[key]).reshape(()).item())


def _precomputed_named_selection(
    archive: Mapping[str, np.ndarray], prefix: str, kind: str, source: str
) -> Optional[GallerySelection]:
    fields = {
        "best_paired": best_paired_field_names,
        "median_paired": median_paired_field_names,
        "best_cross": best_cross_field_names,
    }[kind](prefix)
    required_count = 10 if kind != "best_cross" else 9
    if not all(key in archive for key in fields[:required_count]):
        return None
    if kind == "best_cross":
        return GallerySelection(
            nearest_train_a=np.asarray(archive[fields[5]], dtype=np.float32),
            generated_a=np.asarray(archive[fields[3]], dtype=np.float32),
            generated_b=np.asarray(archive[fields[4]], dtype=np.float32),
            nearest_train_b=np.asarray(archive[fields[6]], dtype=np.float32),
            index_a=_scalar(archive, fields[0], int),
            index_b=_scalar(archive, fields[1], int),
            similarity=_scalar(archive, fields[2], float),
            nearest_train_index_a=_scalar(archive, fields[7], int),
            nearest_train_index_b=_scalar(archive, fields[8], int),
            source=source,
        )
    index = _scalar(archive, fields[0], int)
    return GallerySelection(
        nearest_train_a=np.asarray(archive[fields[6]], dtype=np.float32),
        generated_a=np.asarray(archive[fields[4]], dtype=np.float32),
        generated_b=np.asarray(archive[fields[5]], dtype=np.float32),
        nearest_train_b=np.asarray(archive[fields[7]], dtype=np.float32),
        index_a=index,
        index_b=index,
        similarity=_scalar(archive, fields[3], float),
        nearest_train_index_a=_scalar(archive, fields[8], int),
        nearest_train_index_b=_scalar(archive, fields[9], int),
        source=source,
    )


def _partition_raw_index(data_dir: Path, group: str, n: int, local_index: int) -> int:
    with np.load(data_dir / "paper_partitions.npz", allow_pickle=False) as archive:
        indices = np.asarray(
            archive["group_a_raw_indices" if group == "A" else "group_b_raw_indices"],
            dtype=np.int64,
        )
    subset = indices[:n] if group == "A" else indices[-n:]
    if local_index < 0 or local_index >= subset.size:
        raise IndexError(f"Nearest index {local_index} is outside group {group} subset N={n}.")
    return int(subset[local_index])


def load_partition_image(data_dir: Path, group: str, n: int, local_index: int) -> np.ndarray:
    root = Path(data_dir).expanduser().resolve()
    raw_index = _partition_raw_index(root, group, n, local_index)
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    storage, divisor, cursor = str(manifest["storage"]), float(manifest["decode_divisor"]), 0
    for info in manifest["shards"]:
        count = int(info["count"])
        if cursor <= raw_index < cursor + count:
            path, offset = root / str(info["filename"]), raw_index - cursor
            if storage == "npz_compressed":
                with np.load(path, allow_pickle=False) as archive:
                    image = np.asarray(archive["gray_rgb_sum"][offset], dtype=np.float32)
            elif storage == "npy":
                image = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)[offset], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported prepared storage: {storage}")
            return (image / divisor).reshape(-1)
        cursor += count
    raise IndexError(f"Raw image index {raw_index} was not found in manifest shards.")


def _selection_from_full_arrays(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    *,
    data_dir: Optional[Path],
    n: Optional[int],
    pair_mode: str,
    max_similarity_elements: int,
) -> Optional[GallerySelection]:
    stem = f"{prefix}_" if prefix else ""
    sample_a_key, sample_b_key = f"{stem}sample_a", f"{stem}sample_b"
    nearest_a_key, nearest_b_key = f"{stem}nearest_index_a", f"{stem}nearest_index_b"
    if not all(key in archive for key in (sample_a_key, sample_b_key, nearest_a_key, nearest_b_key)):
        return None
    if data_dir is None or n is None:
        raise ValueError("Full generated arrays require --data-dir and N to recover nearest training images.")
    if pair_mode == "best_cross_set":
        selected = best_cross_set_centered_cosine(
            archive[sample_a_key], archive[sample_b_key],
            max_similarity_elements=max_similarity_elements,
        )
        index_a, index_b, similarity = selected.index_a, selected.index_b, selected.similarity
        source = "full_generated_arrays"
    elif pair_mode == "median_paired_seed":
        selected = median_paired_seed_centered_cosine(
            archive[sample_a_key], archive[sample_b_key], block_size=max_similarity_elements
        )
        index_a = index_b = selected.index
        similarity, source = selected.similarity, "full_generated_arrays_median_paired_seed"
    else:
        selected = best_paired_seed_centered_cosine(
            archive[sample_a_key], archive[sample_b_key], block_size=max_similarity_elements
        )
        index_a = index_b = selected.index
        similarity, source = selected.similarity, "full_generated_arrays_best_paired_seed"
    nearest_a = np.asarray(archive[nearest_a_key], dtype=np.int64).reshape(-1)
    nearest_b = np.asarray(archive[nearest_b_key], dtype=np.int64).reshape(-1)
    train_index_a, train_index_b = int(nearest_a[index_a]), int(nearest_b[index_b])
    return GallerySelection(
        nearest_train_a=load_partition_image(data_dir, "A", n, train_index_a),
        generated_a=np.asarray(archive[sample_a_key][index_a], dtype=np.float32),
        generated_b=np.asarray(archive[sample_b_key][index_b], dtype=np.float32),
        nearest_train_b=load_partition_image(data_dir, "B", n, train_index_b),
        index_a=index_a,
        index_b=index_b,
        similarity=similarity,
        nearest_train_index_a=train_index_a,
        nearest_train_index_b=train_index_b,
        source=source,
    )


def _gallery_position(indices: np.ndarray, requested_index: int) -> int:
    positions = np.flatnonzero(indices == requested_index)
    if positions.size == 0:
        raise IndexError(f"Gallery index {requested_index} is unavailable; saved indices are {indices.tolist()}.")
    return int(positions[0])


def _selection_from_gallery(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    *,
    pair_mode: str,
    gallery_index: int,
    max_similarity_elements: int,
) -> GallerySelection:
    stem = f"{prefix}_" if prefix else ""
    indices_key = f"{stem}gallery_global_index"
    sample_a_key, sample_b_key = f"{stem}gallery_sample_a", f"{stem}gallery_sample_b"
    nearest_a_key, nearest_b_key = f"{stem}gallery_nearest_a", f"{stem}gallery_nearest_b"
    required = (indices_key, sample_a_key, sample_b_key, nearest_a_key, nearest_b_key)
    if not all(key in archive for key in required):
        raise KeyError(f"No compatible gallery arrays found for prefix {prefix!r}.")
    global_indices = np.asarray(archive[indices_key], dtype=np.int64)
    if pair_mode == "same_seed":
        local_a = local_b = _gallery_position(global_indices, gallery_index)
        similarity = centered_cosine_pair(archive[sample_a_key][local_a], archive[sample_b_key][local_b])
        source = "same_seed_gallery_index"
    elif pair_mode == "best_paired_seed":
        warnings.warn("Using only saved gallery seeds for best-paired selection.", RuntimeWarning, stacklevel=3)
        selected = best_paired_seed_centered_cosine(
            archive[sample_a_key], archive[sample_b_key], block_size=max_similarity_elements
        )
        local_a = local_b = selected.index
        similarity, source = selected.similarity, "limited_gallery_best_paired_seed"
    elif pair_mode == "median_paired_seed":
        warnings.warn("Using only saved gallery seeds for median-paired selection.", RuntimeWarning, stacklevel=3)
        selected = median_paired_seed_centered_cosine(
            archive[sample_a_key], archive[sample_b_key], block_size=max_similarity_elements
        )
        local_a = local_b = selected.index
        similarity, source = selected.similarity, "limited_gallery_median_paired_seed"
    else:
        warnings.warn("Using only saved gallery seeds for cross-set selection.", RuntimeWarning, stacklevel=3)
        selected = best_cross_set_centered_cosine(
            archive[sample_a_key], archive[sample_b_key],
            max_similarity_elements=max_similarity_elements,
        )
        local_a, local_b, similarity = selected.index_a, selected.index_b, selected.similarity
        source = "limited_gallery_cross_set"
    global_a, global_b = int(global_indices[local_a]), int(global_indices[local_b])
    nearest_index_a_key, nearest_index_b_key = f"{stem}nearest_index_a", f"{stem}nearest_index_b"
    train_index_a = int(np.asarray(archive[nearest_index_a_key]).reshape(-1)[global_a]) if nearest_index_a_key in archive else -1
    train_index_b = int(np.asarray(archive[nearest_index_b_key]).reshape(-1)[global_b]) if nearest_index_b_key in archive else -1
    return GallerySelection(
        nearest_train_a=np.asarray(archive[nearest_a_key][local_a], dtype=np.float32),
        generated_a=np.asarray(archive[sample_a_key][local_a], dtype=np.float32),
        generated_b=np.asarray(archive[sample_b_key][local_b], dtype=np.float32),
        nearest_train_b=np.asarray(archive[nearest_b_key][local_b], dtype=np.float32),
        index_a=global_a,
        index_b=global_b,
        similarity=float(similarity),
        nearest_train_index_a=train_index_a,
        nearest_train_index_b=train_index_b,
        source=source,
    )


def log_selection(condition: str, selection: GallerySelection) -> None:
    print(
        "[gallery-pair] "
        f"setting={condition} source={selection.source} "
        f"A={selection.index_a} B={selection.index_b} "
        f"centered_cosine={selection.similarity:.6f}"
    )


def select_gallery_from_archive(
    archive: Mapping[str, np.ndarray],
    *,
    prefix: str = "",
    pair_mode: str = "best_paired_seed",
    gallery_index: int = 0,
    condition: str = "unspecified",
    max_similarity_elements: int = DEFAULT_MAX_SIMILARITY_ELEMENTS,
    data_dir: Optional[Path] = None,
    n: Optional[int] = None,
) -> GallerySelection:
    if pair_mode not in ("same_seed", "best_paired_seed", "median_paired_seed", "best_cross_set"):
        raise ValueError(f"Unknown gallery pair mode: {pair_mode}")
    selection: Optional[GallerySelection] = None
    if pair_mode == "best_paired_seed":
        selection = _precomputed_named_selection(archive, prefix, "best_paired", "saved_best_paired_seed")
    elif pair_mode == "median_paired_seed":
        selection = _precomputed_named_selection(archive, prefix, "median_paired", "saved_median_paired_seed")
    elif pair_mode == "best_cross_set":
        selection = _precomputed_named_selection(archive, prefix, "best_cross", "saved_best_cross_set")
    if selection is None and pair_mode != "same_seed":
        selection = _selection_from_full_arrays(
            archive, prefix, data_dir=data_dir, n=n, pair_mode=pair_mode,
            max_similarity_elements=max_similarity_elements,
        )
    if selection is None:
        selection = _selection_from_gallery(
            archive, prefix, pair_mode=pair_mode, gallery_index=gallery_index,
            max_similarity_elements=max_similarity_elements,
        )
    log_selection(condition, selection)
    return selection


class DiskSamplePairStore:
    """Persistent memmap store used to recover exact all-sample selections."""

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace(os.sep, "_")

    @classmethod
    def paths_for(cls, root: Path, name: str) -> Tuple[Path, Path, Path]:
        root, safe_name = Path(root), cls._safe_name(name)
        return (
            root / f"{safe_name}_a.npy",
            root / f"{safe_name}_b.npy",
            root / f"{safe_name}_filled.npy",
        )

    @classmethod
    def exists(cls, root: Path, name: str) -> bool:
        return all(path.is_file() for path in cls.paths_for(root, name))

    def __init__(self, root: Path, name: str, count: int, dimension: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name, self.count, self.dimension = name, int(count), int(dimension)
        self.path_a, self.path_b, self.path_filled = self.paths_for(self.root, name)
        expected_shape = (self.count, self.dimension)
        if self.exists(self.root, name):
            samples_a = np.load(self.path_a, mmap_mode="r+", allow_pickle=False)
            samples_b = np.load(self.path_b, mmap_mode="r+", allow_pickle=False)
            filled = np.load(self.path_filled, allow_pickle=False)
            valid = (
                samples_a.shape == expected_shape and samples_b.shape == expected_shape
                and samples_a.dtype == np.float32 and samples_b.dtype == np.float32
                and filled.shape == (self.count,) and filled.dtype == np.bool_
            )
            if valid:
                self.samples_a, self.samples_b = samples_a, samples_b
                self.filled = np.asarray(filled, dtype=np.bool_)
                return
            del samples_a, samples_b
        for path in (self.path_a, self.path_b, self.path_filled):
            path.unlink(missing_ok=True)
        self.samples_a = np.lib.format.open_memmap(
            self.path_a, mode="w+", dtype=np.float32, shape=expected_shape
        )
        self.samples_b = np.lib.format.open_memmap(
            self.path_b, mode="w+", dtype=np.float32, shape=expected_shape
        )
        self.filled = np.zeros(self.count, dtype=np.bool_)
        self._save_filled()

    def _save_filled(self) -> None:
        temporary = self.path_filled.with_name(self.path_filled.stem + ".tmp.npy")
        np.save(temporary, self.filled, allow_pickle=False)
        os.replace(temporary, self.path_filled)

    def write(self, start: int, generated_a: object, generated_b: object) -> None:
        rows_a, rows_b = _rows(generated_a), _rows(generated_b)
        if rows_a.shape != rows_b.shape or rows_a.shape[1] != self.dimension:
            raise ValueError(
                f"Temporary pair-store shape mismatch: A={rows_a.shape}, B={rows_b.shape}, "
                f"expected dimension={self.dimension}."
            )
        end = start + rows_a.shape[0]
        if start < 0 or end > self.count:
            raise IndexError(f"Temporary pair-store slice {start}:{end} is out of range.")
        self.samples_a[start:end] = rows_a
        self.samples_b[start:end] = rows_b
        self.samples_a.flush()
        self.samples_b.flush()
        self.filled[start:end] = True
        self._save_filled()

    @property
    def complete(self) -> bool:
        return bool(self.filled.all())

    def build_payload(
        self,
        nearest_index_a: object,
        nearest_index_b: object,
        reference_a: object,
        reference_b: object,
        *,
        prefix: str,
        max_similarity_elements: int,
        pair_mode: str = "best_paired_seed",
    ) -> MutableMapping[str, np.ndarray]:
        if not self.complete:
            missing = np.flatnonzero(~self.filled)
            raise RuntimeError(f"Temporary pair store is incomplete; {missing.size} samples are missing.")
        self.samples_a.flush()
        self.samples_b.flush()
        if pair_mode == "best_paired_seed":
            return build_best_paired_payload(
                self.samples_a, self.samples_b, nearest_index_a, nearest_index_b,
                reference_a, reference_b, prefix=prefix,
                block_size=max_similarity_elements,
            )
        if pair_mode == "median_paired_seed":
            return build_median_paired_payload(
                self.samples_a, self.samples_b, nearest_index_a, nearest_index_b,
                reference_a, reference_b, prefix=prefix,
                block_size=max_similarity_elements,
            )
        if pair_mode == "best_cross_set":
            return build_best_cross_payload(
                self.samples_a, self.samples_b, nearest_index_a, nearest_index_b,
                reference_a, reference_b, prefix=prefix,
                max_similarity_elements=max_similarity_elements,
            )
        raise ValueError(f"Unsupported pair_mode for compact payload: {pair_mode}")

    def close(self, *, remove: bool = True) -> None:
        if hasattr(self, "samples_a"):
            self.samples_a.flush()
            self.samples_b.flush()
            del self.samples_a
            del self.samples_b
        if remove:
            for path in (self.path_a, self.path_b, self.path_filled):
                path.unlink(missing_ok=True)


def remove_empty_temp_dir(path: Path) -> None:
    root = Path(path)
    if root.is_dir():
        try:
            root.rmdir()
        except OSError:
            pass
