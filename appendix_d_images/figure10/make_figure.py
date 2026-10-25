"""Build the compact appendix figure for extended MLP training.

The figure uses the four fixed Gaussian sources already evaluated at the
paper checkpoint (200k), an intermediate checkpoint (500k), and the final
approximately 982k checkpoint.  No contrast enhancement is applied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
STAGES = SOURCE / "intermediate" / "intermediate_stages_20261018"
FILES = {
    "200k": STAGES / "baseline_200k" / "samples.npz",
    "500k": STAGES / "step_500k" / "samples.npz",
    "final": SOURCE / "final_samples.npz",
}
OUT = ROOT / "output"

BLUE = "#2166AC"
TEXT = "#202733"
MUTED = "#66717F"
RULE = "#D9DEE5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[int]]]:
    loaded = {name: np.load(path) for name, path in FILES.items()}
    try:
        seeds = loaded["200k"]["seeds"]
        assert np.array_equal(seeds, np.arange(4))
        for stage in ("500k", "final"):
            assert np.array_equal(loaded[stage]["seeds"], seeds)
            assert np.array_equal(loaded[stage]["noise"], loaded["200k"]["noise"])

        images: dict[str, dict[str, np.ndarray]] = {}
        steps: dict[str, list[int]] = {}
        for n in ("10000", "98023"):
            key = f"N{n}_s1_output"
            step_key = f"N{n}_s1_step"
            images[n] = {
                stage: np.asarray(loaded[stage][key], dtype=np.float64).reshape(4, 80, 80)
                for stage in FILES
            }
            steps[n] = [int(loaded[stage][step_key]) for stage in FILES]
            for stage in FILES:
                x = images[n][stage]
                assert x.shape == (4, 80, 80) and np.isfinite(x).all()
                assert x.min() >= 0.0 and x.max() <= 1.0
        return images, steps
    finally:
        for z in loaded.values():
            z.close()


def metrics(images: dict[str, dict[str, np.ndarray]], steps: dict[str, list[int]]) -> dict:
    result = {
        "display": "Endpoint readout clipped to [0,1]; gray colormap; no contrast enhancement",
        "group": "S1",
        "weighting": "EMA",
        "fixed_source_seeds": [0, 1, 2, 3],
        "source_files": {name: {"path": str(path), "sha256": sha256(path)}
                         for name, path in FILES.items()},
        "panels": {},
    }
    for n in ("10000", "98023"):
        final = images[n]["final"].reshape(4, -1)
        panel = {"steps": steps[n], "comparisons_to_final": {}}
        for stage in ("200k", "500k"):
            x = images[n][stage].reshape(4, -1)
            rms = np.sqrt(np.mean((x - final) ** 2, axis=1))
            corr = np.array([np.corrcoef(x[i], final[i])[0, 1] for i in range(4)])
            panel["comparisons_to_final"][stage] = {
                "per_source_pixel_rms": rms.tolist(),
                "mean_pixel_rms": float(rms.mean()),
                "per_source_pixel_correlation": corr.tolist(),
                "mean_pixel_correlation": float(corr.mean()),
            }
        result["panels"][n] = panel
    return result


def draw(images: dict[str, dict[str, np.ndarray]], steps: dict[str, list[int]]) -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "text.color": TEXT,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(6.75, 2.50))
    image_w = 0.083
    image_h = 6.75 * image_w / 2.50
    col_gap = 0.006
    row_gap = 0.014
    left_starts = (0.108, 0.555)
    bottom = 0.027
    row_y = {
        "final": bottom,
        "500k": bottom + image_h + row_gap,
        "200k": bottom + 2 * (image_h + row_gap),
    }

    fig.text(0.014, 0.918, r"MLP EMA, $S_1$", ha="left", va="center",
             fontsize=10.2, fontweight="semibold")
    fig.text(0.082, 0.799, "Seed", ha="right", va="center", color=MUTED)

    row_labels = {
        "200k": ("200k", "main"),
        "500k": ("500k", ""),
        "final": (r"$\approx$982k", "final"),
    }
    for stage in ("200k", "500k", "final"):
        y = row_y[stage] + image_h / 2
        primary, secondary = row_labels[stage]
        fig.text(0.088, y + (0.014 if secondary else 0), primary,
                 ha="right", va="center", fontsize=9.1,
                 color=BLUE if stage in ("200k", "final") else TEXT,
                 fontweight="semibold" if stage in ("200k", "final") else "normal")
        if secondary:
            fig.text(0.088, y - 0.029, secondary, ha="right", va="center",
                     fontsize=6.5, color=MUTED)

    for panel_idx, n in enumerate(("10000", "98023")):
        x0 = left_starts[panel_idx]
        center = x0 + (4 * image_w + 3 * col_gap) / 2
        fig.text(center, 0.918, rf"({chr(97 + panel_idx)})  $N={int(n):,}$",
                 ha="center", va="center", fontsize=10.8, fontweight="semibold")
        for col in range(4):
            x = x0 + col * (image_w + col_gap)
            fig.text(x + image_w / 2, 0.799, str(col), ha="center", va="center",
                     fontsize=8.3, color=MUTED)
            for stage in ("200k", "500k", "final"):
                ax = fig.add_axes([x, row_y[stage], image_w, image_h])
                ax.imshow(images[n][stage][col], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", rasterized=True)
                ax.set_axis_off()
                if stage == "final":
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.55)
                        spine.set_edgecolor(BLUE)

        # A short rule separates the headers from the three image rows.
        fig.lines.append(plt.Line2D([x0, x0 + 4 * image_w + 3 * col_gap],
                                    [0.766, 0.766], transform=fig.transFigure,
                                    color=RULE, lw=0.65))

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"appendix_mlp_training_saturation.{suffix}",
                    dpi=360, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    images, steps = load_inputs()
    summary = metrics(images, steps)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "appendix_mlp_training_saturation_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    draw(images, steps)
    for n, panel in summary["panels"].items():
        vals = panel["comparisons_to_final"]
        print(
            f"N={n}: corr(200k,final)={vals['200k']['mean_pixel_correlation']:.4f}; "
            f"corr(500k,final)={vals['500k']['mean_pixel_correlation']:.4f}; "
            f"RMS(500k,final)={vals['500k']['mean_pixel_rms']:.4f}"
        )


if __name__ == "__main__":
    main()
