"""Offline figures for paired short-window intervention runs.

Only NumPy, Matplotlib and the saved run artifacts are required.  Pixel distances
describe changes to samples; they are not estimates of perceptual quality or
generalization.  Run ``python -m celeba_nn_stage.window_figures RUN_DIR`` to
regenerate the figures without loading a model or using a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


_DIRECTIONS = (
    ("nn_emp_nn", "NN > empirical > NN", "full_nn"),
    ("emp_nn_emp", "Empirical > NN > empirical", "full_emp"),
)
_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}


def _title(manifest: dict[str, Any], text: str) -> str:
    return ("DEMO / synthetic data - " if manifest.get("demo", False) else "") + text


def _save(fig, directory, stem):
    path = directory / f"{stem}.png"
    try:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)
    return [path]



def _window(method: dict[str, Any]) -> tuple[float, float] | None:
    window = method.get("window")
    if window is None:
        return None
    return float(window[0]), float(window[1])


def _label(method: dict[str, Any]) -> str:
    return str(method.get("label", method["id"]))


def _image(vector: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(vector).reshape(shape)


def _show_image(ax: plt.Axes, vector: np.ndarray, shape: tuple[int, ...]) -> None:
    # Clipping is solely for display.  All measurements use saved, unclipped data.
    ax.imshow(np.clip(_image(vector, shape), 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _method_groups(methods: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    by_id = {method["id"]: method for method in methods}
    groups = []
    included = set()
    for direction, label, fallback in _DIRECTIONS:
        interventions = sorted(
            [method for method in methods if method.get("direction") == direction],
            key=lambda method: sum(_window(method) or (0, 0)),
        )
        baseline_id = interventions[0].get("baseline", fallback) if interventions else fallback
        group = ([by_id[baseline_id]] if baseline_id in by_id else []) + interventions
        if group:
            groups.append((direction, label, group))
            included.update(method["id"] for method in group)
    other = [method for method in methods if method["id"] not in included]
    if other:
        groups.append(("other", "Other methods", other))
    return groups


def _galleries(
    run: Path,
    directory: Path,
    manifest: dict[str, Any],
    shape: tuple[int, ...],
    gallery_seeds: int,
) -> list[Path]:
    paths = []
    for direction, label, group in _method_groups(manifest["methods"]):
        for page, start in enumerate(range(0, len(group), 4), 1):
            subset = group[start : start + 4]
            data = []
            for method in subset:
                with np.load(run / "samples" / f"{method['id']}.npz", allow_pickle=False) as saved:
                    data.append({key: np.asarray(saved[key])[:gallery_seeds] for key in
                                 ("final", "nearest_s1", "nearest_s2")})
            count = min(gallery_seeds, *(len(item[key]) for item in data for key in item))
            if count == 0:
                continue
            fig, axes = plt.subplots(3 * len(subset), count, squeeze=False,
                                     figsize=(2.05 * count + 2.7, 1.7 * 3 * len(subset) + 1.0))
            fig.subplots_adjust(left=0.23, right=0.99, bottom=0.045, top=0.935, hspace=0.10, wspace=0.04)
            for row, (method, samples) in enumerate(zip(subset, data)):
                for offset, (key, row_label) in enumerate((
                    ("final", _label(method) + "\nFinal output"),
                    ("nearest_s1", "Nearest S1 image"),
                    ("nearest_s2", "Nearest S2 image"),
                )):
                    for seed in range(count):
                        ax = axes[3 * row + offset, seed]
                        _show_image(ax, samples[key][seed], shape)
                        if row == 0 and offset == 0:
                            ax.set_title(f"Paired seed {seed}")
                    axes[3 * row + offset, 0].set_ylabel(
                        "\n".join(textwrap.fill(line, 24) for line in row_label.splitlines()),
                        rotation=0, ha="right", va="center", labelpad=9,
                    )
            fig.suptitle(_title(manifest, f"Final samples: {label} (page {page})"), y=0.985, fontsize=13)
            fig.text(0.5, 0.012, "Same initial noise in each column. Nearest images use raw pixel RMSE; display clipped to [0, 1].",
                     ha="center", fontsize=9)
            paths.extend(_save(fig, directory, f"final_gallery_{direction}_{page:02d}"))
    return paths


def _metrics(run: Path) -> dict[str, dict[int, dict[str, str]]]:
    grouped: dict[str, dict[int, dict[str, str]]] = {}
    with (run / "metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            seed = int(row["seed_index"])
            if seed in grouped.setdefault(method, {}):
                raise ValueError(f"Duplicate paired seed {seed} for method {method} in metrics.csv")
            grouped[method][seed] = row
    return grouped


def _number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value else float("nan")


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return (float("nan"),) * 3
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    # Resampling whole paired-seed differences retains the pairing.  Batching
    # bounds memory even for runs containing many samples.
    resampled = []
    for _ in range(20):
        indices = rng.integers(0, len(values), size=(100, len(values)))
        resampled.append(np.mean(values[indices], axis=1))
    low, high = np.quantile(np.concatenate(resampled), [0.025, 0.975])
    return mean, float(low), float(high)


def _paired_effects(run: Path, directory: Path, manifest: dict[str, Any]) -> list[Path]:
    metrics = _metrics(run)
    paths = []
    rng = np.random.default_rng(20260906)
    panels = (
        ("delta_final_rmse", "Final-output change", "Paired final-output RMSE"),
        ("delta_window_rmse", "Change at window exit", "Paired window-exit RMSE"),
        ("nearest_s1_rmse", "Nearest-S1 distance change", "Nearest-S1 RMSE minus baseline"),
    )
    for direction, label, group in _method_groups(manifest["methods"]):
        interventions = [method for method in group if _window(method) is not None]
        if not interventions:
            continue
        # Different window widths are different experimental series.
        widths: dict[float, list[dict[str, Any]]] = {}
        for method in interventions:
            a, b = _window(method)  # type: ignore[misc]
            widths.setdefault(round(b - a, 10), []).append(method)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        sample_counts = set()
        baseline_names = set()
        for width, series in sorted(widths.items()):
            for ax, (metric, title, ylabel) in zip(axes, panels):
                points = []
                for method in sorted(series, key=lambda item: sum(_window(item) or (0, 0))):
                    baseline_id = method["baseline"]
                    baseline_names.add(baseline_id)
                    rows = metrics.get(method["id"], {})
                    baseline = metrics.get(baseline_id, {})
                    shared_seeds = sorted(rows.keys() & baseline.keys())
                    values = np.asarray([
                        _number(rows[seed], metric) - (
                            _number(baseline[seed], metric) if metric == "nearest_s1_rmse" else 0.0
                        ) for seed in shared_seeds
                    ], dtype=float)
                    sample_counts.add(int(np.isfinite(values).sum()))
                    points.append((sum(_window(method)) / 2, *_bootstrap_mean(values, rng)))  # type: ignore[arg-type]
                values = np.asarray(points)
                x, means, low, high = values.T
                line = ax.plot(x, means, "o-", markersize=4, label=f"Window width {width:g}")[0]
                color = line.get_color()
                valid_ci = np.isfinite(low) & np.isfinite(high)
                ax.fill_between(x, low, high, where=valid_ci, color=color, alpha=0.20)
                # A visible interval is still required for a series of one window.
                ax.vlines(x[valid_ci], low[valid_ci], high[valid_ci], color=color, linewidth=1.1)
                ax.set(title=title, xlabel="Window center t", ylabel=ylabel)
                ax.grid(alpha=0.2)
                ax.axhline(0, color="0.5", linewidth=0.7)
                if metric != "nearest_s1_rmse":
                    ax.set_ylim(bottom=0)
        axes[0].legend(loc="best")
        n_label = ", ".join(str(n) for n in sorted(sample_counts))
        fig.suptitle(_title(manifest, f"Paired pixel changes: {label}"), fontsize=13)
        fig.text(0.5, 0.012,
                 f"Means and 95% paired-seed bootstrap intervals (2,000 resamples; n = {n_label}; CI omitted for n < 2).\n"
                 f"Baseline: {', '.join(sorted(baseline_names))}. Pixel distances measure change, not image quality.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.12, 1, 0.94))
        paths.extend(_save(fig, directory, f"paired_effect_{direction}"))
    return paths


def _detail_selection(methods, requested):
    by_id = {m["id"]: m for m in methods}
    if requested is not None:
        unknown = set(requested) - by_id.keys()
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        return [by_id[key] for key in dict.fromkeys(requested)]
    return list(methods)



def _boundary_marks(axes: np.ndarray, times: np.ndarray, window: tuple[float, float] | None) -> None:
    if window is None or not len(times):
        return
    for boundary, symbol in zip(window, ("a", "b")):
        if times[0] <= boundary <= times[-1]:
            right = int(np.searchsorted(times, boundary, side="left"))
            # The line falls between probe columns; no unrecorded state is implied.
            right = min(right, len(times) - 1)
            for row in axes:
                row[right].axvline(-0.5, color="#d88000", linewidth=2)
            axes[0, right].annotate(f"{symbol}={boundary:g}", (0, 1.23), xycoords="axes fraction",
                                    color="#995900", fontsize=9, ha="left")


def _clean_panels(
    run: Path,
    directory: Path,
    manifest: dict[str, Any],
    shape: tuple[int, ...],
    requested: list[str] | None,
    detail_seeds: int,
) -> list[Path]:
    selected = _detail_selection(manifest["methods"], requested)
    traces = []
    difference_limit = 0.0
    for method in selected:
        trace_path = run / "diagnostics" / f"{method['id']}.npz"
        if not trace_path.exists():
            continue
        with np.load(trace_path, allow_pickle=False) as saved:
            data = {key: np.asarray(saved[key])[:detail_seeds] for key in
                    ("state", "clean_nn", "clean_emp", "difference")}
            data["times"] = np.asarray(saved["times"])
        finite = np.abs(data["difference"][np.isfinite(data["difference"])])
        if finite.size:
            difference_limit = max(difference_limit, float(np.max(finite)))
        traces.append((method, data))
    difference_limit = max(difference_limit, 1e-12)
    norm = Normalize(vmin=-difference_limit, vmax=difference_limit)
    paths = []
    for method, data in traces:
        times = data["times"]
        if not len(times):
            continue
        window = _window(method)
        window_label = "No intervention window" if window is None else f"Window boundaries: a={window[0]:g}, b={window[1]:g}"
        for seed in range(len(data["state"])):
            for page, start in enumerate(range(0, len(times), 7), 1):
                indices = np.arange(start, min(start + 7, len(times)))
                page_times = times[indices]
                fig, axes = plt.subplots(4, len(indices), squeeze=False,
                                         figsize=(2.0 * len(indices) + 2.7, 8.5))
                fig.subplots_adjust(left=0.16, right=0.94, bottom=0.13, top=0.80, hspace=0.07, wspace=0.04)
                for row, (key, label) in enumerate((
                    ("state", "Actual state x(t)"),
                    ("clean_nn", "NN clean prediction"),
                    ("clean_emp", "Empirical clean prediction"),
                    ("difference", "Signed NN - empirical"),
                )):
                    for column, index in enumerate(indices):
                        ax = axes[row, column]
                        if key == "difference":
                            ax.imshow(_image(data[key][seed, index], shape), cmap="RdBu_r", norm=norm)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                        else:
                            _show_image(ax, data[key][seed, index], shape)
                        if row == 0:
                            ax.set_title(f"t = {times[index]:g}", pad=9)
                    axes[row, 0].set_ylabel(textwrap.fill(label, 21), rotation=0, ha="right", va="center", labelpad=9)
                _boundary_marks(axes, page_times, window)
                colorbar_axis = fig.add_axes([0.955, 0.16, 0.011, 0.13])
                colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), cax=colorbar_axis)
                colorbar.ax.tick_params(labelsize=9)
                fig.suptitle(_title(manifest, f"Same-state clean predictions: {_label(method)} | seed {seed} | page {page}"),
                             fontsize=13, y=0.98)
                fig.text(0.5, 0.91, window_label + "\nBoth predictors are evaluated on the same saved x(t).", ha="center", fontsize=10)
                fig.text(0.5, 0.022,
                         f"Detail seed {seed} belongs to the diagnostic subset of {manifest.get('diagnostic_seeds', '?')} / "
                         f"{manifest.get('num_seeds', '?')} output seeds.\n"
                         "Clean predictions are denoised estimates, not actual future trajectory states.\n"
                         "Image rows: display clipped to [0, 1]. Signed difference: raw values, common scale across all detail panels.\n"
                         "Orange lines locate window boundaries between available probes; exact boundaries are stated above.",
                         ha="center", fontsize=9)
                paths.extend(_save(fig, directory, f"clean_prediction_{method['id']}_seed{seed:03d}_{page:02d}"))
    return paths


def _quantiles(values: np.ndarray) -> np.ndarray:
    result = np.full((3, values.shape[1]), np.nan)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if len(finite):
            result[:, index] = np.quantile(finite, [0.25, 0.5, 0.75])
    return result


def _trajectory_panels(
    run: Path,
    directory: Path,
    manifest: dict[str, Any],
    shape: tuple[int, ...],
    requested: list[str] | None,
    detail_seeds: int,
) -> list[Path]:
    """Show actual paired states, including the saved final output at t=1."""
    selected = [method for method in _detail_selection(manifest["methods"], requested)
                if _window(method) is not None]
    traces = []
    difference_limit = 0.0
    for method in selected:
        baseline_id = method["baseline"]
        path = run / "diagnostics" / f"{method['id']}.npz"
        baseline_path = run / "diagnostics" / f"{baseline_id}.npz"
        if not path.exists() or not baseline_path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved, np.load(baseline_path, allow_pickle=False) as baseline:
            times = np.asarray(saved["times"])
            if not np.array_equal(times, baseline["times"]):
                raise ValueError(f"Paired trajectory probe times differ: {method['id']} and {baseline_id}")
            count = min(detail_seeds, len(saved["state"]), len(baseline["state"]))
            states = np.asarray(saved["state"])[:count]
            base_states = np.asarray(baseline["state"])[:count]
        if count == 0:
            continue
        with np.load(run / "samples" / f"{method['id']}.npz", allow_pickle=False) as saved:
            final = np.asarray(saved["final"])[:count]
        with np.load(run / "samples" / f"{baseline_id}.npz", allow_pickle=False) as saved:
            base_final = np.asarray(saved["final"])[:count]
        times = np.append(times, 1.0)
        states = np.concatenate((states, final[:, None, :]), axis=1)
        base_states = np.concatenate((base_states, base_final[:, None, :]), axis=1)
        difference = states - base_states
        finite = np.abs(difference[np.isfinite(difference)])
        if finite.size:
            difference_limit = max(difference_limit, float(np.max(finite)))
        traces.append((method, times, base_states, states, difference))
    difference_limit = max(difference_limit, 1e-12)
    norm = Normalize(vmin=-difference_limit, vmax=difference_limit)
    paths = []
    for method, times, baseline, states, difference in traces:
        window = _window(method)
        for seed in range(len(states)):
            for page, start in enumerate(range(0, len(times), 7), 1):
                indices = np.arange(start, min(start + 7, len(times)))
                fig, axes = plt.subplots(3, len(indices), squeeze=False,
                                         figsize=(2.0 * len(indices) + 2.7, 7.0))
                fig.subplots_adjust(left=0.16, right=0.94, bottom=0.16, top=0.79, hspace=0.07, wspace=0.04)
                for row, (array, label) in enumerate((
                    (baseline, "Baseline state"),
                    (states, "Intervention state"),
                    (difference, "Signed intervention - baseline"),
                )):
                    for column, index in enumerate(indices):
                        ax = axes[row, column]
                        if row == 2:
                            ax.imshow(_image(array[seed, index], shape), cmap="RdBu_r", norm=norm)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                        else:
                            _show_image(ax, array[seed, index], shape)
                        if row == 0:
                            title = "t = 1 (final)" if times[index] == 1 else f"t = {times[index]:g}"
                            ax.set_title(title, pad=9)
                    axes[row, 0].set_ylabel(textwrap.fill(label, 21), rotation=0,
                                            ha="right", va="center", labelpad=9)
                _boundary_marks(axes, times[indices], window)
                colorbar_axis = fig.add_axes([0.955, 0.19, 0.011, 0.14])
                colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), cax=colorbar_axis)
                colorbar.ax.tick_params(labelsize=9)
                fig.suptitle(_title(manifest, f"Actual paired trajectories: {_label(method)} | seed {seed} | page {page}"),
                             fontsize=13, y=0.98)
                fig.text(0.5, 0.89,
                         f"Baseline: {method['baseline']}; window boundaries a={window[0]:g}, b={window[1]:g}.\n"
                         "The two runs share initial noise, time grid and the field outside the window.",
                         ha="center", fontsize=10)
                fig.text(0.5, 0.030,
                         f"Detail seed {seed} belongs to the diagnostic subset of {manifest.get('diagnostic_seeds', '?')} / "
                         f"{manifest.get('num_seeds', '?')} output seeds.\n"
                         "Rows show actual saved states; t=1 is the saved final output. Images are displayed clipped to [0, 1].\n"
                         "Signed differences use raw states and one common scale across paired-trajectory panels.\n"
                         "Orange lines mark window boundaries. Paired differences measure change, not image quality.",
                         ha="center", fontsize=9)
                paths.extend(_save(fig, directory, f"paired_trajectory_{method['id']}_seed{seed:03d}_{page:02d}"))
    return paths


def _diagnostic_curves(run: Path, directory: Path, manifest: dict[str, Any]) -> list[Path]:
    fields = (
        ("denoiser_rmse", "Same-state clean-prediction RMSE"),
        ("velocity_rmse", "Same-state velocity RMSE"),
        ("cosine", "Same-state velocity cosine similarity"),
        ("ess", "log10 empirical ESS"),
        ("top_probability", "Largest empirical weight"),
    )
    summaries = {}
    maxima = {key: 0.0 for key, _ in fields}
    for method in manifest["methods"]:
        path = run / "diagnostics" / f"{method['id']}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            times = np.asarray(saved["times"])
            if not len(times) or not len(saved["denoiser_rmse"]):
                continue
            summary = {"times": times, "count": len(saved["denoiser_rmse"])}
            for key, _ in fields:
                values = np.asarray(saved[key], dtype=float)
                if key == "ess":
                    values = np.where(values > 0, np.log10(np.maximum(values, 1e-300)), np.nan)
                quantiles = _quantiles(values)
                summary[key] = quantiles
                finite = quantiles[np.isfinite(quantiles)]
                if len(finite):
                    maxima[key] = max(maxima[key], float(np.max(finite)))
            summaries[method["id"]] = summary
    paths = []
    for direction, label, group in _method_groups(manifest["methods"]):
        group = [method for method in group if method["id"] in summaries]
        for page, start in enumerate(range(0, len(group), 5), 1):
            subset = group[start : start + 5]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            handles = []
            for index, method in enumerate(subset):
                summary = summaries[method["id"]]
                color = plt.get_cmap("tab10")(index)
                for ax, (key, title) in zip(axes.flat, fields):
                    low, median, high = summary[key]
                    line = ax.plot(summary["times"], median, color=color, linewidth=1.5,
                                   label=f"{_label(method)} (n={summary['count']})")[0]
                    ax.fill_between(summary["times"], low, high, color=color, alpha=0.12)
                    ax.set(title=title, xlabel="Time t")
                    ax.grid(alpha=0.2)
                    ax.set_xlim(0, 1)
                    if key == "cosine":
                        ax.set_ylim(-1.03, 1.03)
                    elif key == "top_probability":
                        ax.set_ylim(0, 1.03)
                    else:
                        ax.set_ylim(0, max(1e-8, maxima[key] * 1.06))
                    window = _window(method)
                    if window is not None:
                        for boundary in window:
                            ax.axvline(boundary, color=color, linestyle=":", alpha=0.5, linewidth=0.8)
                    if key == "denoiser_rmse":
                        handles.append(line)
            axes[1, 2].axis("off")
            axes[1, 2].legend(handles=handles, loc="upper left", frameon=False)
            axes[1, 2].text(0.03, 0.40,
                            f"Diagnostic subset: first {manifest.get('diagnostic_seeds', '?')} / "
                            f"{manifest.get('num_seeds', '?')} output seeds.\n"
                            "Lines: median across diagnostic seeds.\nBands: 25th-75th percentiles, not CIs.\n"
                            "Dotted lines: each method's window boundaries.\n"
                            "Both fields are evaluated at the same x(t).\n"
                            "Non-finite observations are omitted.\n"
                            "Clean prediction and velocity are distinct;\nRMSE here measures pixel / vector differences.",
                            transform=axes[1, 2].transAxes, fontsize=9, va="top", linespacing=1.5)
            fig.suptitle(_title(manifest, f"Trajectory diagnostics: {label} (page {page})"), fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            paths.extend(_save(fig, directory, f"diagnostic_curves_{direction}_{page:02d}"))
    return paths


def plot_run(
    out_dir: str | Path,
    gallery_seeds: int = 4,
    detail_methods: list[str] | None = None,
    detail_seeds: int = 2,
) -> list[Path]:
    """Write PNG/PDF figures into ``out_dir/figures`` and return their paths.

    ``detail_methods=None`` selects both full baselines and the intervention
    closest to t=0.5 in each direction.  An empty list disables clean-prediction
    panels.  ``detail_seeds=0`` also disables those panels.  Bootstrap intervals
    are descriptive, pointwise percentile intervals over matched saved seeds.
    """
    run = Path(out_dir).expanduser().resolve()
    with (run / "manifest.json").open(encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    shape = tuple(int(value) for value in manifest["image_shape"])
    if len(shape) != 2 or any(value < 1 for value in shape):
        raise ValueError("manifest.image_shape must contain positive [height, width] dimensions")
    if gallery_seeds < 0 or detail_seeds < 0:
        raise ValueError("gallery_seeds and detail_seeds must be nonnegative")
    directory = run / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(_STYLE):
        if gallery_seeds:
            paths.extend(_galleries(run, directory, manifest, shape, gallery_seeds))
        paths.extend(_paired_effects(run, directory, manifest))
        if detail_seeds:
            paths.extend(_clean_panels(run, directory, manifest, shape, detail_methods, detail_seeds))
            paths.extend(_trajectory_panels(run, directory, manifest, shape, detail_methods, detail_seeds))
        paths.extend(_diagnostic_curves(run, directory, manifest))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Saved run directory containing manifest.json")
    parser.add_argument("--gallery-seeds", type=int, default=4)
    parser.add_argument("--detail-seeds", type=int, default=2)
    parser.add_argument("--detail-methods", nargs="*", default=None, metavar="METHOD")
    args = parser.parse_args()
    paths = plot_run(args.out_dir, args.gallery_seeds, args.detail_methods, args.detail_seeds)
    print(f"Wrote {len(paths)} figure files to {args.out_dir.resolve() / 'figures'}")


if __name__ == "__main__":
    main()
