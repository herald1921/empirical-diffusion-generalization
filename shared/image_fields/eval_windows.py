#!/usr/bin/env python3
"""One entry point for paired short-window interventions and their figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from celeba_nn_stage.window_experiment import (
    Window, build_grid, integrate_grid, run_window, compare_clean,
)
from celeba_nn_stage.sampling import make_fixed_noise

DEFAULT_RUN = Path("/root/shared-nvme/experiment-real-data/runs/celeba_NN_stage_S1_t_uniform")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "checkpoints/checkpoint_step_200000.pt")
    p.add_argument("--base-repo", type=Path, default=Path("/root/shared-nvme/experiment-mnist/celeba80_cloud_bundle"))
    p.add_argument("--data-dir", type=Path, default=Path("/root/shared-nvme/experiment-real-data/datasets/celeba80_author_order"))
    p.add_argument("--windows", default="0:0.1,0.1:0.2,0.2:0.3,0.3:0.4,0.4:0.5,0.5:0.6,0.6:0.7,0.7:0.8,0.8:0.9,0.9:1")
    p.add_argument("--probe-times", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98")
    p.add_argument("--num-seeds", type=int, default=64)
    p.add_argument("--diagnostic-seeds", type=int, default=8)
    p.add_argument("--gallery-seeds", type=int, default=4)
    p.add_argument("--noise-seed", type=int, default=20260810)
    p.add_argument("--solver-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--reference-chunk-size", type=int, default=4096)
    p.add_argument("--weights", choices=("ema", "raw"), default="ema")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true", help="Reuse only completed method/batch files with an identical manifest.")
    p.add_argument("--plot-only", action="store_true", help="Rebuild figures from saved outputs, without loading a model.")
    p.add_argument("--demo", action="store_true", help="CPU synthetic shapes for software/figure checks; NOT CelebA results.")
    return p.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def tensor_hash(value):
    digest = hashlib.sha256()
    for part in value.split(1024):
        digest.update(part.contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def demo_fields():
    """Finite cloud versus a continuous Gaussian shape distribution, only for QA."""
    size = 24
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    features = torch.stack((torch.exp(-4 * (xx**2 + yy**2)), xx * torch.exp(-3 * (xx**2 + yy**2)), yy * torch.exp(-3 * (xx**2 + yy**2)))).reshape(3, -1).T
    basis, _ = torch.linalg.qr(features)
    mu = (0.35 + 0.25 * torch.exp(-3 * (xx**2 + yy**2))).flatten()
    g = torch.Generator().manual_seed(751)
    def draw(n):
        return mu + 1.1 * torch.randn(n, 3, generator=g) @ basis.T + 0.025 * torch.randn(n, size * size, generator=g)
    s1, s2 = draw(24), draw(24)

    def empirical(x, t):
        weights = torch.softmax(-torch.cdist(x, float(t) * s1).square() / (2 * (1 - float(t))**2), dim=1)
        mean = weights @ s1
        return (mean - x) / (1 - float(t)), {"mean": mean, "ess": weights.square().sum(1).reciprocal(), "top_probability": weights.max(1).values}

    def neural(x, t):
        # Analytic continuous target, not a trained neural network.
        z = x - float(t) * mu
        parallel = (z @ basis) @ basis.T
        eig_small, eig_large = 0.025**2, 0.025**2 + 1.1**2
        def coefficient(eig):
            return (t * eig - (1-t)) / ((1-t)**2 + t*t*eig)
        return mu + coefficient(eig_small) * (z - parallel) + coefficient(eig_large) * parallel
    return s1, s2, neural, empirical, [size, size], {"target": "synthetic Gaussian shapes; NN label means analytic continuous-target field in DEMO only"}


def real_fields(args, device):
    from celeba_nn_stage.data import load_full_groups_cpu, _load_core
    from celeba_nn_stage.model import build_model_from_checkpoint
    from celeba_nn_stage.empirical_t import empirical_velocity_t
    s1, s2, info, _ = load_full_groups_cpu(data_dir=args.data_dir, base_repo=args.base_repo)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.weights == "ema" and "ema_model" not in payload:
        raise ValueError("EMA requested but checkpoint has no EMA weights")
    old = payload.get("dataset_info", {})
    for key, actual in (("count", len(s1)), ("dimension", s1.shape[1]), ("decode_divisor", info.decode_divisor)):
        if key in old and old[key] != actual:
            raise ValueError(f"Checkpoint data mismatch: {key}: {old[key]} != {actual}")
    model = build_model_from_checkpoint(payload, device, use_ema=args.weights == "ema")
    ref = s1.to(device)
    core = _load_core(args.base_repo)
    norms = core.reference_norms(ref, args.reference_chunk_size)
    mean = ref.mean(0)
    def neural(x, t):
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16" and device.type == "cuda"):
            return model(x, float(t)).float()
    def empirical(x, t):
        return empirical_velocity_t(core=core, x_t=x, reference=ref, reference_norms=norms, reference_mean=mean,
                                    t=float(t), query_batch_size=args.batch_size, reference_chunk_size=args.reference_chunk_size)
    provenance = {"checkpoint_sha256": file_hash(args.checkpoint), "checkpoint_step": payload.get("step"),
                  "core_sha256": file_hash(args.base_repo / "celeba80_theory_core.py"), "dataset_info": info.__dict__}
    return s1, s2, neural, empirical, [80, 80], provenance


@torch.inference_mode()
def nearest(q, reference, device, chunk):
    """Full-reference pixel RMSE; recompute chosen distances directly for accuracy."""
    q = q.to(device)
    best = torch.full((len(q),), float("inf"), device=device)
    indices = torch.zeros(len(q), dtype=torch.long, device=device)
    for start in range(0, len(reference), chunk):
        r = reference[start:start+chunk].to(device)
        # Avoid cancellation in ||q||^2+||r||^2-2q.r when testing near copies.
        dist = torch.cdist(q, r, compute_mode="donot_use_mm_for_euclid_dist").square()
        value, local = dist.min(1)
        better = value < best
        indices[better] = start + local[better]
        best = torch.minimum(best, value)
    selected = reference[indices.cpu()]
    rmse = (q.cpu() - selected).square().mean(1).sqrt()
    return rmse.numpy(), selected.numpy()


def methods_for(windows):
    methods = [dict(id="full_nn", label="Full NN", direction="baseline_nn", window=None, baseline="full_nn"),
               dict(id="full_emp", label="Full empirical", direction="baseline_emp", window=None, baseline="full_emp")]
    for direction, base, label in (("nn_emp_nn", "full_nn", "NN / empirical / NN"), ("emp_nn_emp", "full_emp", "Empirical / NN / empirical")):
        for w in windows:
            methods.append(dict(id=f"{direction}_{w.key}", label=f"{label} [{w.start:g}, {w.end:g})", direction=direction,
                                window=[w.start, w.end], baseline=base))
    return methods


def main():
    args = arguments()
    out = args.out_dir.expanduser().resolve()
    if args.plot_only:
        from celeba_nn_stage.window_figures import plot_run
        plot_run(out, gallery_seeds=args.gallery_seeds)
        return
    for key in ("num_seeds", "diagnostic_seeds", "gallery_seeds", "solver_steps", "batch_size", "reference_chunk_size"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    windows = [Window(*map(float, item.split(":"))) for item in args.windows.split(",")]
    if len({w.key for w in windows}) != len(windows):
        raise ValueError("Duplicate windows")
    probes = sorted(set(round(float(x), 12) for x in args.probe_times.split(",")))
    if not probes or any(not np.isfinite(t) or not 0 <= t < 1 for t in probes):
        raise ValueError("Clean-prediction probes must satisfy 0 <= t < 1; no empirical prediction at t=1")
    # Include all intervention boundaries in diagnostics, except the singular endpoint.
    probes = sorted(set(probes + [v for w in windows for v in (w.start, w.end) if v < 1]))
    captures = sorted(set(probes + [0., 1.] + [v for w in windows for v in (w.start, w.end)]))
    grid = build_grid(args.solver_steps, windows, captures)
    device = torch.device("cpu" if args.demo else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --demo for a CPU software check, or run the real experiment on the training server.")
    torch.manual_seed(args.noise_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.demo:
        torch.set_num_threads(1)
    print("[setup] loading fields and original data", flush=True)
    s1, s2, nn, empirical, shape, provenance = demo_fields() if args.demo else real_fields(args, device)
    fixed = make_fixed_noise(args.num_seeds, s1.shape[1], args.noise_seed)
    methods = methods_for(windows)
    k, g = min(args.diagnostic_seeds, args.num_seeds), min(args.gallery_seeds, args.num_seeds)
    source_files = [Path(__file__), ROOT / "src/celeba_nn_stage/window_experiment.py", ROOT / "src/celeba_nn_stage/empirical_t.py", ROOT / "src/celeba_nn_stage/model.py", ROOT / "src/celeba_nn_stage/data.py", ROOT / "src/celeba_nn_stage/sampling.py"]
    manifest = dict(schema_version=1, demo=args.demo, image_shape=shape, num_seeds=args.num_seeds, diagnostic_seeds=k,
                    gallery_seeds=g, noise_seed=args.noise_seed, solver="explicit midpoint RK2", grid=list(grid),
                    probe_times=probes, methods=methods, weights=args.weights, precision=args.precision,
                    batch_size=args.batch_size, reference_chunk_size=args.reference_chunk_size,
                    provenance=provenance, s1_sha256=tensor_hash(s1), s2_sha256=tensor_hash(s2),
                    s1_count=len(s1), s2_count=len(s2), source_hashes={p.name: file_hash(p) for p in source_files},
                    diagnostic_selection="first K fixed Gaussian seeds; same subset for every method",
                    metrics_note="Pixel proximity and trajectory changes are NOT standalone quality or generalization scores.",
                    runtime=dict(torch=torch.__version__, device=str(device)))
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        if not args.resume:
            raise FileExistsError("Output exists. Use a new --out-dir, or --resume with identical configuration.")
        if json.loads(manifest_file.read_text(encoding="utf-8")) != json.loads(json.dumps(manifest)):
            raise ValueError("Resume manifest mismatch: data, source, model or configuration changed. Use a new output directory.")
    else:
        if any(out.iterdir()):
            raise FileExistsError("Output is not empty and has no manifest. Use a new directory.")
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    all_metrics = []
    for method in methods:
        mid = method["id"]
        print(f"[method] {method['label']}", flush=True)
        finals, nearest1, nearest2, diag_parts = [], [], [], []
        for start in range(0, args.num_seeds, args.batch_size):
            end = min(args.num_seeds, start + args.batch_size)
            cache = out / "batches" / mid / f"{start:05d}_{end:05d}.npz"
            if cache.exists():
                print(f"  [{start}:{end}] verified run cache", flush=True)
                with np.load(cache) as z:
                    part = {key: z[key] for key in z.files}
            else:
                print(f"  [{start}:{end}] integrate + diagnose", flush=True)
                if method["window"] is None:
                    field = nn if mid == "full_nn" else lambda x, t: empirical(x, t)[0]
                    final, caps = integrate_grid(field, fixed[start:end].to(device), grid, captures)
                    base_final, window_delta = final.cpu(), torch.zeros(end-start)
                else:
                    w = Window(*method["window"])
                    base_path = out / "batches" / method["baseline"] / cache.name
                    with np.load(base_path) as z:
                        base_caps = torch.from_numpy(z["captures"])
                        base_final = torch.from_numpy(z["final"])
                    at_start = base_caps[:, captures.index(w.start)].to(device)
                    final, caps = run_window(nn, lambda x, t: empirical(x, t)[0], at_start, grid, w, method["direction"], captures)
                    for index, t in enumerate(captures):
                        if t < w.start:
                            caps[t] = base_caps[:, index]
                    window_delta = (caps[w.end] - base_caps[:, captures.index(w.end)]).square().mean(1).sqrt()
                final_cpu = final.cpu()
                r1, image1 = nearest(final_cpu, s1, device, args.reference_chunk_size)
                r2, image2 = nearest(final_cpu, s2, device, args.reference_chunk_size)
                part = dict(final=final_cpu.numpy(), captures=torch.stack([caps[t] for t in captures], dim=1).numpy(),
                            nearest_s1=image1, nearest_s2=image2, nearest_s1_rmse=r1, nearest_s2_rmse=r2,
                            delta_final_rmse=(final_cpu-base_final).square().mean(1).sqrt().numpy(),
                            delta_window_rmse=window_delta.numpy(), output_outside_fraction=((final_cpu<0)|(final_cpu>1)).float().mean(1).numpy())
                kd = max(0, min(end, k)-start)
                if kd:
                    records = [compare_clean(nn, empirical, caps[t][:kd].to(device), t) for t in probes]
                    for key in records[0]:
                        part["diag_"+key] = torch.stack([r[key] for r in records], dim=1).numpy()
                save_npz(cache, **part)
            finals.append(part["final"])
            nearest1.append(part["nearest_s1"])
            nearest2.append(part["nearest_s2"])
            if "diag_state" in part:
                diag_parts.append({key[5:]: val for key, val in part.items() if key.startswith("diag_")})
            metric_keys = ("nearest_s1_rmse", "nearest_s2_rmse", "delta_final_rmse", "delta_window_rmse", "output_outside_fraction")
            for j in range(end-start):
                all_metrics.append(dict(method=mid, seed_index=start+j, **{key: float(part[key][j]) for key in metric_keys}))
        save_npz(out / "samples" / f"{mid}.npz", final=np.concatenate(finals), source_noise=fixed.numpy(),
                 nearest_s1=np.concatenate(nearest1)[:g], nearest_s2=np.concatenate(nearest2)[:g])
        save_npz(out / "diagnostics" / f"{mid}.npz", times=np.array(probes),
                 **{key: np.concatenate([d[key] for d in diag_parts]) for key in diag_parts[0]})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    from celeba_nn_stage.window_figures import plot_run
    plot_run(out, gallery_seeds=g)
    (out / "COMPLETE").write_text("All methods and figures completed.\n", encoding="utf-8")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
