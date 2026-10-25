"""Reformat the frozen Figure 7 results, without resampling or changing inputs."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
manifest = json.loads((DATA / 'main/manifest.json').read_text())
validated = ROOT / 'validated_data'
validation = json.loads((validated / 'provenance.json').read_text())
assert hashlib.sha256((validated / 'metrics.csv').read_bytes()).hexdigest() == validation['corrected_metrics_sha256']
rows = list(csv.DictReader((validated / 'metrics.csv').open()))
perceptual = list(csv.DictReader((DATA / 'perceptual/metrics.csv').open()))
snap = np.load(DATA / 'snapshots/states.npz')
nearest = np.load(DATA / 'perceptual/gallery.npz')
plt.rcParams.update({
    'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.linewidth': .65, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
})
W, H = 7.2, 4.23
fig = plt.figure(figsize=(W, H))
blue, orange = '#0072B2', '#D55E00'

def label(x, y, text, **kwargs):
    return fig.text(x / W, y / H, text, **kwargs)

def axis(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])

# Two close pairs per seed, with successively wider gaps between pairs and seeds.
size, pair_gap, condition_gap, seed_gap = .625, .025, .13, .27
pair_width = 2 * size + pair_gap
seed_width = 2 * pair_width + condition_gap
x_starts = [1.45, 1.45 + seed_width + seed_gap]
col_offsets = [0, size + pair_gap, pair_width + condition_gap,
               pair_width + condition_gap + size + pair_gap]
for x0 in x_starts:
    for offset, heading in [(0, 'Original'), (pair_width + condition_gap, 'Replaced')]:
        label(x0 + offset + pair_width / 2, 4.115, heading,
              ha='center', fontsize=8.8)
        fig.add_artist(plt.Line2D(
            [(x0 + offset) / W, (x0 + offset + pair_width) / W],
            [4.045 / H, 4.045 / H], transform=fig.transFigure,
            color='#aeb2b5', lw=.5))
    for offset, heading in zip(col_offsets, ['Output', 'Nearest train'] * 2):
        label(x0 + offset + size / 2, 3.915, heading, ha='center', fontsize=7.4)

cases = [
    ('(a) Early replacement', r'Empirical $\to$ neural', '[0.10, 0.40)',
     'emp', 'emp_nn_pulse_0.1_0.3'),
    ('(b) Late replacement', r'Empirical $\to$ neural', '[0.69, 0.99)',
     'emp', 'emp_nn_pulse_0.69_0.3'),
    ('(c) Late replacement', r'Neural $\to$ empirical', '[0.69, 0.99)',
     'nn', 'nn_emp_pulse_0.69_0.3'),
]
for p, (title, direction, interval, base, method) in enumerate(cases):
    y = 3.20 - p * .675
    center = y + size / 2
    label(.08, center + .13, title, fontsize=8.5, fontweight='bold', va='center')
    label(.08, center - .025, direction, fontsize=8.2, va='center')
    label(.08, center - .18, interval, fontsize=8, va='center')
    for x0, seed in zip(x_starts, [2, 1]):
        items = [snap[f'baseline_{base}__0.99'][seed],
                 nearest[f'full_{base}__nearest_lpips'][seed],
                 snap[method + '__0.99'][seed],
                 nearest[method + '__nearest_lpips'][seed]]
        for offset, im in zip(col_offsets, items):
            ax = axis(x0 + offset, y, size, size)
            ax.imshow(im.reshape(80, 80), cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#777777')
                spine.set_linewidth(.3)

boot = np.random.default_rng(20261018).integers(0, 64, (5000, 64))
axes = []
curve_audit = {}
for k in range(2):
    ax = axis(.51 + k * 3.63, .35, 2.83, .98)
    axes.append(ax)
    highest = 0
    for base, color, legend, marker in [
        ('emp', blue, r'Empirical $\to$ neural', 'o'),
        ('nn', orange, r'Neural $\to$ empirical', 's'),
    ]:
        methods = sorted([m for m in manifest['methods']
                          if m['kind'] == 'pulse' and m['base'] == base
                          and m['width'] == .3], key=lambda m: m['start'])
        key = 'change_from_' + base if k == 0 else 'lpips_train'
        source = rows if k == 0 else perceptual
        values = np.array([[float(r[key]) for r in sorted(source, key=lambda r: int(r['seed']))
                            if r['method'] == m['id']] for m in methods])
        assert values.shape == (len(methods), 64)
        lo, hi = np.quantile(values[:, boot].mean(2), [.025, .975], axis=1)
        x = [m['start'] for m in methods]
        ax.fill_between(x, lo, hi, color=color, alpha=.14, lw=0)
        ax.plot(x, values.mean(1), color=color, marker=marker, lw=1.3,
                ms=2.5, ls='-' if base == 'emp' else '--', label=legend)
        curve_audit[f'{k}_{base}'] = {'x': x, 'mean': values.mean(1).tolist()}
        highest = max(highest, float(hi.max()))
    ax.set_title('(d) Change in final output' if k == 0 else '(e) Proximity to training images',
                 fontsize=9, fontweight='bold', pad=7)
    ax.set_xlabel('Replacement start $a$', labelpad=3)
    ax.set_xlim(-.01, .705)
    ax.set_xticks([0, .2, .4, .6, .69])
    ax.grid(alpha=.14, lw=.5)
    ax.set_ylabel('RMSE from original' if k == 0 else 'Nearest-training LPIPS', labelpad=5)
    if k == 1:
        for base, text in [('emp', 'Pure empirical'), ('nn', 'Pure neural')]:
            baseline = np.mean([float(r['lpips_train']) for r in perceptual
                                if r['method'] == 'full_' + base])
            ax.axhline(baseline, color='#737373', lw=.8, ls=':', zorder=0)
            ax.annotate(text, xy=(.025, baseline), xytext=(0, 6),
                        textcoords='offset points', va='bottom', fontsize=7.2,
                        color='#555555')
            highest = max(highest, baseline)
    ax.set_ylim(0, highest * (1.32 if k == 1 else 1.14))

fig.legend(*axes[0].get_legend_handles_labels(), loc='center',
           bbox_to_anchor=(.54, 1.68 / H), ncol=2, frameon=False,
           fontsize=8.2, borderaxespad=0, handlelength=1.8,
           handletextpad=.7, columnspacing=2.5)
for ext in ['png', 'pdf']:
    fig.savefig(ROOT / f'figure7_faces_first.{ext}', dpi=300,
                bbox_inches='tight', pad_inches=.06)
(ROOT / 'figure7_faces_first_audit.json').write_text(json.dumps({
    'displayed_seed_indices': [2, 1], 'curve_sample_count': 64,
    'pair_gap_inches': pair_gap, 'condition_gap_inches': condition_gap,
    'seed_gap_inches': seed_gap, 'curves': curve_audit,
}, indent=2))
plt.close(fig)
