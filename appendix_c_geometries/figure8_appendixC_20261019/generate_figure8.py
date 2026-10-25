"""Build Appendix Figure 8 from the paper's synthetic protocol.

The four rows use the same deterministic N=16 designs, empirical Tweedie
map, normalized noise eta=sigma/alpha, and distributional diagnostics as the
main text.  Betti numbers are computed for the finite unions of balls with
radius r=c_{D,0.05} eta.  The shaded regions report computed Betti agreement;
they are not theorem certificates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.spatial.distance import cdist
from scipy.special import ndtri, softmax
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / "appendix_jmlr_revision_20261018" / "evidence_bundle"
META = json.loads((EVIDENCE / "appendix_data" / "synthetic_metadata.json").read_text())
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

ETA = np.array([.008, .015, .025, .04, .06, .08, .11, .15,
                .20, .28, .38, .52, .72, 1., 1.4, 2.])
DELTA = .05
N = 16
EVAL_COUNT = 512

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#17836B"
INK = "#253642"
GRAY = "#89969F"
LIGHT_GRAY = "#E5EAED"
PALE_GREEN = "#DCECE5"
PALE_BLUE = "#DCECF6"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 7.4,
    "axes.labelsize": 7.4,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.7,
    "pdf.fonttype": 42,
    "axes.edgecolor": "#AEB8BE",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#5F6E78",
    "ytick.color": "#5F6E78",
    "axes.linewidth": .55,
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c_delta(dim: int) -> float:
    return float(np.sqrt(dim) + np.sqrt(2 * np.log(1 / DELTA)))


def angles(u: np.ndarray, amplitude: float) -> np.ndarray:
    theta = 2 * np.pi * u
    for _ in range(15):
        theta -= (theta + amplitude * np.sin(theta) - 2 * np.pi * u) / (
            1 + amplitude * np.cos(theta)
        )
    return theta


def eight_lut(count: int = 131073):
    t = np.linspace(0, 2 * np.pi, count)
    raw = np.c_[np.sin(t), np.sin(2 * t)]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    arc /= arc[-1]
    probe_t = np.interp((np.arange(65536) + .5) / 65536, arc, t)
    scale = 1 / np.sqrt(np.mean(np.sin(probe_t) ** 2 + np.sin(2 * probe_t) ** 2))
    return t, arc, scale


EIGHT_T, EIGHT_ARC, EIGHT_SCALE = eight_lut()


def eight_points(count: int) -> np.ndarray:
    u = (np.arange(count) + .5) / count
    t = np.interp(u, EIGHT_ARC, EIGHT_T)
    return EIGHT_SCALE * np.c_[np.sin(t), np.sin(2 * t)]


def training(shape: str) -> np.ndarray:
    if shape in {"ellipse", "sphere"}:
        key = f"{shape}_N16_a{0.0 if shape == 'sphere' else 0.1:.1f}_design_r0"
        path = EVIDENCE / "designs" / key / "training.npy"
        assert sha256(path) == META[key]["training_sha256"]
        return np.load(path)
    if shape == "segment":
        return np.c_[np.sqrt(3) * (2 * (np.arange(N) + .5) / N - 1), np.zeros(N)]
    if shape == "eight":
        return eight_points(N)
    raise ValueError(shape)


def target(shape: str, count: int, seed: int = 777) -> np.ndarray:
    if shape == "ellipse":
        t = angles((np.arange(count) + .5) / count, .1)
        return np.c_[2 * np.cos(t), np.sin(t)]
    if shape == "sphere":
        u = qmc.Sobol(2, scramble=True, seed=seed).random_base2(int(np.log2(count)))
        z, theta = 1 - 2 * u[:, 0], 2 * np.pi * u[:, 1]
        r = np.sqrt(1 - z * z)
        return 2 * np.c_[r * np.cos(theta), r * np.sin(theta), z]
    if shape == "segment":
        return np.c_[np.sqrt(3) * (2 * (np.arange(count) + .5) / count - 1), np.zeros(count)]
    if shape == "eight":
        return eight_points(count)
    raise ValueError(shape)


def project_ellipse(x: np.ndarray) -> np.ndarray:
    ax2 = np.array([4., 1.])
    y = np.abs(x)
    off_axis = y[:, 1] > 1e-14
    ans = np.zeros_like(x)
    if off_axis.any():
        v = y[off_axis]
        lo = np.full(len(v), np.nextafter(-1., 0.))
        hi = np.maximum(1., 2 * np.linalg.norm(v, axis=1))
        for _ in range(85):
            mid = .5 * (lo + hi)
            residual = np.sum(ax2 * v * v / (ax2 + mid[:, None]) ** 2, axis=1) - 1
            lo = np.where(residual > 0, mid, lo)
            hi = np.where(residual > 0, hi, mid)
        ans[off_axis] = ax2 * v / (ax2 + (.5 * (lo + hi))[:, None])
    if (~off_axis).any():
        u = y[~off_axis, 0]
        px = np.minimum(2., 4 * u / 3)
        ans[~off_axis] = np.c_[px, np.sqrt(np.maximum(0., 1 - px * px / 4))]
    return ans * np.where(x < 0, -1., 1.)


def project_eight(x: np.ndarray) -> np.ndarray:
    # A dense global start resolves the two branches at the crossing; Newton
    # iterations then locate the nearest stationary point on the smooth pieces.
    mesh_t = np.linspace(0, 2 * np.pi, 32768, endpoint=False)
    mesh = EIGHT_SCALE * np.c_[np.sin(mesh_t), np.sin(2 * mesh_t)]
    _, idx = cKDTree(mesh).query(x)
    t = mesh_t[idx]
    for _ in range(12):
        p = EIGHT_SCALE * np.c_[np.sin(t), np.sin(2 * t)]
        dp = EIGHT_SCALE * np.c_[np.cos(t), 2 * np.cos(2 * t)]
        ddp = EIGHT_SCALE * np.c_[-np.sin(t), -4 * np.sin(2 * t)]
        grad = np.sum((p - x) * dp, axis=1)
        hess = np.sum(dp * dp + (p - x) * ddp, axis=1)
        step = np.divide(grad, hess, out=np.zeros_like(grad), where=np.abs(hess) > 1e-12)
        t = np.mod(t - np.clip(step, -.08, .08), 2 * np.pi)
    p = EIGHT_SCALE * np.c_[np.sin(t), np.sin(2 * t)]
    # Include the crossing explicitly because it is a second local branch.
    use_crossing = np.sum(x * x, axis=1) < np.sum((x - p) ** 2, axis=1)
    p[use_crossing] = 0.
    return p


def project(x: np.ndarray, shape: str) -> np.ndarray:
    if shape == "ellipse":
        return project_ellipse(x)
    if shape == "sphere":
        radius = np.linalg.norm(x, axis=1, keepdims=True)
        ans = 2 * x / np.maximum(radius, 1e-300)
        ans[radius[:, 0] == 0] = np.array([2., 0., 0.])
        return ans
    if shape == "segment":
        return np.c_[np.clip(x[:, 0], -np.sqrt(3), np.sqrt(3)), np.zeros(len(x))]
    if shape == "eight":
        return project_eight(x)
    raise ValueError(shape)


def noisy_nodes(train: np.ndarray, count: int, seed: int = 2000):
    per_atom = count // len(train)
    assert per_atom * len(train) == count and per_atom & (per_atom - 1) == 0
    blocks = []
    for i in range(len(train)):
        u = qmc.Sobol(train.shape[1], scramble=True, seed=seed + 101 * i).random_base2(
            int(np.log2(per_atom)) - 1
        )
        g = ndtri(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
        blocks.append(np.r_[g, -g])
    return np.repeat(train, per_atom, axis=0), np.vstack(blocks)


def w2_equal(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x) == len(y)
    cost = cdist(x, y, "sqeuclidean")
    row, col = linear_sum_assignment(cost)
    return float(np.sqrt(np.mean(cost[row, col])))


def distribution_series(shape: str):
    train = training(shape)
    truth = target(shape, EVAL_COUNT)
    z, g = noisy_nodes(train, EVAL_COUNT)
    repeated = np.repeat(train, EVAL_COUNT // len(train), axis=0)
    endpoint = w2_equal(repeated, truth)
    rows = []
    for eta in ETA:
        x = z + eta * g
        post = softmax(-cdist(x, train, "sqeuclidean") / (2 * eta * eta), axis=1) @ train
        mapped = project(post, shape)
        rows.append({
            "eta": float(eta),
            "w2": w2_equal(post, truth),
            "wt": w2_equal(mapped, truth),
            "wn": float(np.sqrt(np.mean(np.sum((post - mapped) ** 2, axis=1)))),
        })
    return rows, endpoint


def circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab, bc, ca = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
    u, v = b - a, c - a
    twice_area = abs(u[0] * v[1] - u[1] * v[0])
    if twice_area < 1e-13:
        return np.inf
    return float(ab * bc * ca / (2 * twice_area))


def planar_alpha(shape: str, points: np.ndarray):
    if shape == "segment":
        order = np.argsort(points[:, 0])
        edges = [(int(order[i]), int(order[i + 1]),
                  float(np.linalg.norm(points[order[i + 1]] - points[order[i]]) / 2))
                 for i in range(len(points) - 1)]
        return edges, []

    tri = Delaunay(points)
    triangles = []
    adjacent = {}
    for simplex in tri.simplices:
        i, j, k = map(int, simplex)
        radius = circumradius(points[i], points[j], points[k])
        triangles.append((i, j, k, radius))
        for u, v in [(i, j), (j, k), (k, i)]:
            e = tuple(sorted((u, v)))
            adjacent.setdefault(e, []).append(radius)

    edges = []
    for (i, j), radii in adjacent.items():
        half = float(np.linalg.norm(points[i] - points[j]) / 2)
        midpoint = (points[i] + points[j]) / 2
        empty_diametral = np.min(np.linalg.norm(points - midpoint, axis=1)[
            [k for k in range(len(points)) if k not in (i, j)]
        ]) >= half - 1e-11
        alpha = half if empty_diametral else min(radii)
        edges.append((i, j, float(alpha)))
    return edges, triangles


def betti_from_alpha(n: int, edges, triangles, radius: float):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    active_edges = [(i, j) for i, j, a in edges if a <= radius + 1e-12]
    for i, j in active_edges:
        union(i, j)
    b0 = len({find(i) for i in range(n)})
    faces = sum(a <= radius + 1e-12 for *_, a in triangles)
    chi = n - len(active_edges) + faces
    b1 = int(round(b0 - chi))
    return (b0, b1)


def metadata_betti(meta: dict, radius: float, dimensions: int):
    return tuple(
        sum(b <= radius + 1e-12 and (d is None or radius < d - 1e-12)
            for b, d in meta["intervals_radius"].get(str(k), []))
        for k in range(dimensions)
    )


def topology_record(shape: str):
    train = training(shape)
    c = c_delta(train.shape[1])
    wanted = {
        "ellipse": (1, 1),
        "sphere": (1, 0, 1),
        "segment": (1, 0),
        "eight": (1, 2),
    }[shape]

    if shape in {"ellipse", "sphere"}:
        key = f"{shape}_N16_a{0.0 if shape == 'sphere' else 0.1:.1f}_design_r0"
        meta = META[key]
        critical = {0.}
        for intervals in meta["intervals_radius"].values():
            for birth, death in intervals:
                critical.add(float(birth))
                if death is not None:
                    critical.add(float(death))
        betti = lambda r: metadata_betti(meta, r, len(wanted))
        coverage = meta["cover_radius_or_upper_bound"]
        reach = meta["reach"]
    else:
        edges, triangles = planar_alpha(shape, train)
        critical = {0.} | {a for *_, a in edges} | {a for *_, a in triangles}
        betti = lambda r: betti_from_alpha(len(train), edges, triangles, r)
        coverage = None
        reach = None

    plot_max = float(ETA.max() * c)
    boundaries = sorted(x for x in critical if np.isfinite(x) and x <= plot_max)
    boundaries = sorted(set(boundaries + [plot_max]))
    spans = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mid = (lo + hi) / 2
        if betti(mid) == wanted:
            if spans and abs(spans[-1][1] - lo) < 1e-8:
                spans[-1][1] = hi
            else:
                spans.append([lo, hi])
    # Coalesce machine-scale splits created by cocircular alpha events.
    merged = []
    for lo, hi in spans:
        if merged and lo - merged[-1][1] < 1e-6:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    spans = merged

    eta_dense = np.geomspace(ETA.min(), ETA.max(), 500)
    betti_dense = [betti(c * e) for e in eta_dense]
    return {
        "c_delta": c,
        "wanted": wanted,
        "spans_radius": spans,
        "spans_eta": [[lo / c, hi / c] for lo, hi in spans],
        "eta_dense": eta_dense,
        "betti_dense": betti_dense,
        "coverage": coverage,
        "reach": reach,
    }


def stages(shape: str, topo: dict):
    fixed = {
        "ellipse": (.38, .15, .025),
        "sphere": (.72, .38, .11),
        "segment": (.38, .06, .015),
    }
    if shape in fixed:
        return fixed[shape]
    spans = topo["spans_eta"]
    available = [z for z in spans if np.any((ETA > z[0]) & (ETA < z[1]))]
    longest = max(available or spans, key=lambda z: z[1] - z[0])
    midpoint = np.sqrt(max(longest[0], 1e-6) * longest[1])
    inside = ETA[(ETA > longest[0]) & (ETA < longest[1])]
    mid = float(inside[np.argmin(abs(np.log(inside / midpoint)))]) if len(inside) else float(midpoint)
    high_candidates = ETA[ETA > longest[1] * 1.25]
    high = float(high_candidates[0] if len(high_candidates) else ETA[-1])
    low_candidates = ETA[ETA < longest[0] * .65]
    low = float(low_candidates[-1] if len(low_candidates) else ETA[0])
    return high, mid, low


def support_curve(shape: str, count: int = 800):
    if shape == "ellipse":
        t = np.linspace(0, 2 * np.pi, count)
        return np.c_[2 * np.cos(t), np.sin(t)]
    if shape == "sphere":
        t = np.linspace(0, 2 * np.pi, count)
        return np.c_[2 * np.cos(t), 2 * np.sin(t)]
    if shape == "segment":
        return np.c_[np.linspace(-np.sqrt(3), np.sqrt(3), count), np.zeros(count)]
    if shape == "eight":
        return eight_points(count)
    raise ValueError(shape)


def set_geometry_frame(ax, shape: str):
    limits = {
        "ellipse": ((-2.75, 2.75), (-1.65, 1.65)),
        "sphere": ((-2.65, 2.65), (-2.65, 2.65)),
        "segment": ((-2.05, 2.05), (-1.15, 1.15)),
        "eight": ((-1.65, 1.65), (-1.65, 1.65)),
    }[shape]
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal")
    ax.axis("off")


def draw_support(ax, shape: str, train: np.ndarray, label: tuple[str, str]):
    set_geometry_frame(ax, shape)
    curve = support_curve(shape)
    if shape == "sphere":
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(2 * np.cos(t), 2 * np.sin(t), color=GRAY, lw=.7)
        for scale in [-.55, 0, .55]:
            ax.plot(2 * np.cos(t) * np.sqrt(1 - scale ** 2),
                    np.full_like(t, 2 * scale), color=GRAY, lw=.35, alpha=.55)
        ax.plot([-2, 2], [0, 0], color=GRAY, lw=.35, alpha=.55)
        ax.scatter(train[:, 0], train[:, 1], s=7, c=INK, zorder=4, linewidths=0)
    else:
        ax.plot(curve[:, 0], curve[:, 1], color=GRAY, lw=.75, zorder=1)
        ax.scatter(train[:, 0], train[:, 1], s=7, c=INK, zorder=4, linewidths=0)
    ax.text(.01, .98, label[0], transform=ax.transAxes, ha="left", va="top",
            fontsize=8.8, fontweight="bold", color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=.25))
    ax.text(.01, .82, label[1], transform=ax.transAxes, ha="left", va="top",
            fontsize=7.0, color=GRAY,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=.2))


def draw_neighborhood(ax, shape: str, train: np.ndarray, eta: float, color: str):
    set_geometry_frame(ax, shape)
    radius = c_delta(train.shape[1]) * eta
    if shape == "sphere":
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(2 * np.cos(t), 2 * np.sin(t), color=GRAY, lw=.55, ls=(0, (2, 2)), alpha=.8)
        for p in train:
            cross = np.sqrt(max(0., radius * radius - p[2] * p[2]))
            if cross > 0:
                ax.add_patch(Circle(p[:2], cross, facecolor=color, edgecolor=color,
                                    lw=.25, alpha=.14))
        ax.scatter(train[:, 0], train[:, 1], s=4, c=INK, zorder=5, linewidths=0)
    else:
        curve = support_curve(shape)
        ax.plot(curve[:, 0], curve[:, 1], color=GRAY, lw=.55,
                ls=(0, (2, 2)), alpha=.8, zorder=1)
        for p in train:
            ax.add_patch(Circle(p[:2], radius, facecolor=color, edgecolor=color,
                                lw=.25, alpha=.14, zorder=2))
        ax.scatter(train[:, 0], train[:, 1], s=4, c=INK, zorder=5, linewidths=0)
    ax.text(.5, -.03, rf"$\eta={eta:g}$", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.7, color=INK)


def draw_distribution(ax, shape: str, rows, endpoint: float, topo: dict, show_xlabel: bool):
    eta = np.array([r["eta"] for r in rows])
    for lo, hi in topo["spans_eta"]:
        lo, hi = max(lo, ETA.min()), min(hi, ETA.max())
        if hi > lo:
            ax.axvspan(lo, hi, color=PALE_GREEN, alpha=.72, lw=0, zorder=-3)

    for field, color, label in [
        ("w2", BLUE, r"$W_2$"),
        ("wt", GREEN, r"$W_T$"),
        ("wn", ORANGE, r"$W_N$"),
    ]:
        values = np.maximum(np.array([r[field] for r in rows]) / endpoint, 1e-4)
        ax.plot(eta, values, color=color, lw=1.25, label=label)
    ax.axhline(1, color=GRAY, lw=.7, ls=(0, (3, 2)), zorder=-1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ETA.max(), ETA.min())
    ax.set_ylim(1e-3, 10)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
    ax.grid(axis="y", color=LIGHT_GRAY, lw=.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([1, .1, .01], ["1", "0.1", "0.01"])
    ax.minorticks_off()
    if show_xlabel:
        ax.set_xlabel(r"Normalized noise $\eta$ (decreasing $\longrightarrow$)", labelpad=1)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"Distance / $S_N$", labelpad=1.5)
    if shape == "segment":
        ax.text(.035, .045, r"$W_N=0$", transform=ax.transAxes, color=ORANGE,
                fontsize=6.6, ha="left", va="bottom")

    beta = np.asarray(topo["betti_dense"], dtype=int)
    eta_dense = topo["eta_dense"]
    second_index = 2 if shape == "sphere" else 1
    ax2 = ax.twinx()
    ax2.step(eta_dense, beta[:, 0], where="mid", color="#71869A", lw=.65,
             ls=(0, (4, 2)), alpha=.9)
    ax2.step(eta_dense, beta[:, second_index], where="mid", color="#71869A", lw=.65,
             ls=(0, (1.2, 1.6)), alpha=.9)
    ax2.set_ylim(-.3, max(3.2, beta[:, 0].max() + .5))
    ticks = [0, 1, 2, int(beta[:, 0].max())]
    ticks = sorted(set(t for t in ticks if t <= beta[:, 0].max()))
    ax2.set_yticks(ticks)
    ax2.tick_params(axis="y", colors="#71869A", pad=1, length=2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#AEB8BE")
    ax2.set_ylabel("Betti number", color="#71869A", labelpad=1.5)


def main():
    shapes = ["ellipse", "sphere", "segment", "eight"]
    labels = {
        "ellipse": ("Ellipse", "closed curve"),
        "sphere": ("Sphere", r"closed surface $2\mathbb{S}^2$"),
        "segment": ("Segment", "boundary"),
        "eight": ("Figure eight", "crossing; outside assumptions"),
    }

    computed = {}
    for shape in shapes:
        print(f"Computing {shape} ...", flush=True)
        topo = topology_record(shape)
        rows, endpoint = distribution_series(shape)
        stage = stages(shape, topo)
        computed[shape] = {
            "topology": topo,
            "distribution": rows,
            "endpoint": endpoint,
            "stages": stage,
        }

    fig = plt.figure(figsize=(7.15, 6.55))
    gs = fig.add_gridspec(
        4, 5, left=.035, right=.975, bottom=.07, top=.87,
        width_ratios=[1.14, 1., 1., 1., 2.45], hspace=.22, wspace=.16
    )

    centers = []
    for j in range(5):
        pos = gs[0, j].get_position(fig)
        centers.append((pos.x0 + pos.x1) / 2)
    headers = ["Support + observations", "High noise", "Intermediate noise",
               "Low noise", "Distributional accuracy"]
    for x, header in zip(centers, headers):
        fig.text(x, .925, header, ha="center", va="center", fontsize=8.5,
                 fontweight="bold", color=INK)
    fig.text(np.mean(centers[1:4]), .975,
             r"Generation direction: normalized noise $\eta$ decreases $\longrightarrow$",
             ha="center", va="center", fontsize=8.0, color=INK)
    fig.add_artist(Line2D([gs[0, 1].get_position(fig).x0,
                           gs[0, 3].get_position(fig).x1], [.953, .953],
                          transform=fig.transFigure, color="#607D94", lw=.65))

    for i, shape in enumerate(shapes):
        train = training(shape)
        draw_support(fig.add_subplot(gs[i, 0]), shape, train, labels[shape])
        high, mid, low = computed[shape]["stages"]
        for j, (eta, color) in enumerate([(high, BLUE), (mid, GREEN), (low, "#4DA3CF")], 1):
            draw_neighborhood(fig.add_subplot(gs[i, j]), shape, train, eta, color)
        draw_distribution(fig.add_subplot(gs[i, 4]), shape,
                          computed[shape]["distribution"], computed[shape]["endpoint"],
                          computed[shape]["topology"], i == len(shapes) - 1)
        if i < len(shapes) - 1:
            y = gs[i, 0].get_position(fig).y0 - .017
            fig.add_artist(Line2D([.025, .985], [y, y], transform=fig.transFigure,
                                  color="#DCE3E7", lw=.45))

    legend = [
        Line2D([], [], color=BLUE, lw=1.3, label=r"$W_2$"),
        Line2D([], [], color=GREEN, lw=1.3, label=r"$W_T$"),
        Line2D([], [], color=ORANGE, lw=1.3, label=r"$W_N$"),
        Line2D([], [], color=GRAY, lw=.8, ls=(0, (3, 2)), label=r"$S_N$"),
        Patch(facecolor=PALE_GREEN, edgecolor="none", label="Betti agreement"),
        Line2D([], [], color="#71869A", lw=.7, ls=(0, (4, 2)), label=r"$\beta_0$"),
        Line2D([], [], color="#71869A", lw=.7, ls=(0, (1.2, 1.6)),
               label=r"target $\beta_k$"),
    ]
    fig.legend(handles=legend, ncol=7, frameon=False, loc="upper right",
               bbox_to_anchor=(.985, .907), handlelength=1.9, columnspacing=.9,
               handletextpad=.35)

    pdf = OUT / "appendix_geometry_four_supports.pdf"
    png = OUT / "appendix_geometry_four_supports.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=.035)
    fig.savefig(png, dpi=260, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)

    serializable = {}
    for shape, entry in computed.items():
        topo = entry["topology"]
        serializable[shape] = {
            "endpoint_SN": entry["endpoint"],
            "stages_eta": list(entry["stages"]),
            "c_delta": topo["c_delta"],
            "target_betti": list(topo["wanted"]),
            "betti_agreement_radius": topo["spans_radius"],
            "betti_agreement_eta": topo["spans_eta"],
            "coverage": topo["coverage"],
            "reach": topo["reach"],
            "distribution": entry["distribution"],
        }
    audit = {
        "protocol": {
            "N": N,
            "delta": DELTA,
            "evaluation_nodes": EVAL_COUNT,
            "eta_grid": ETA.tolist(),
            "transport": "equal-weight exact assignment on deterministic evaluation nodes",
            "map": "empirical posterior mean (Tweedie map)",
            "shading": "computed Betti agreement for the finite ball union; not a theorem certificate",
        },
        "supports": serializable,
        "output_pdf_sha256": sha256(pdf),
    }
    (OUT / "figure8_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(json.dumps({k: {"stages": v["stages_eta"],
                          "betti_eta": v["betti_agreement_eta"],
                          "SN": v["endpoint_SN"]}
                      for k, v in serializable.items()}, indent=2), flush=True)
    print(pdf, flush=True)


if __name__ == "__main__":
    main()
