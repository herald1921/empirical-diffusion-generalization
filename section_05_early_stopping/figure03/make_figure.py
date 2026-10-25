"""Unified Section 5 illustration, replotted from the frozen Figures 3--5 data."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'
SIZE = (6.75, 4.84)
C = {'raw': '#0072B2', 'tw': '#D55E00', 'ink': '#26343D',
     'gray': '#7E8B94', 'support': '#B5BDC2', 'grid': '#E7ECEF'}
plt.rcParams.update({
    'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'text.color': C['ink'], 'axes.labelcolor': C['ink'],
})

def read(name):
    rows = list(csv.DictReader((DATA / name).open(encoding='utf-8')))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

d = read('figure05_curves.csv')
order = np.argsort(d['eta'])[::-1]
d = {k: v[order] for k, v in d.items()}
n = read('figure04_display_normal.csv')
cloud = np.load(DATA / 'inset_data.npz')
inset_audit = json.loads((DATA / 'inset_audit.json').read_text())
P = cloud['training_points']
R = float(cloud['radius'])
SN = float(d['endpoint'][0])
fig = plt.figure(figsize=SIZE)

def axes(x, y, w, h):
    return fig.add_axes([x / SIZE[0], y / SIZE[1], w / SIZE[0], h / SIZE[1]])

def text(x, y, s, **kw):
    return fig.text(x / SIZE[0], y / SIZE[1], s, **kw)

def arrow(x0, y0, x1, y1, color=C['gray'], scale=7):
    fig.add_artist(FancyArrowPatch(
        (x0 / SIZE[0], y0 / SIZE[1]), (x1 / SIZE[0], y1 / SIZE[1]),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=scale,
        lw=.7, color=color))

def exact_row(eta):
    j = int(np.argmin(abs(d['eta'] - eta)))
    assert abs(d['eta'][j] - eta) < 1e-12
    return {k: float(v[j]) for k, v in d.items()}

readouts = {key: exact_row(eta) for key, eta in [('A', .35), ('B', .16)]}
minimum = exact_row(inset_audit['minimum']['eta'])
noisy = exact_row(inset_audit['noisy']['eta'])
assert minimum['raw_wt'] == float(d['raw_wt'].min())
assert readouts['A']['tweedie_w2'] < readouts['A']['raw_w2']
assert readouts['B']['tweedie_w2'] > readouts['B']['raw_w2']
assert all(r['tweedie_w2'] < SN for r in readouts.values())

handles = [Line2D([0], [0], c=C['raw'], lw=1.7, label='Raw'),
           Line2D([0], [0], c=C['tw'], lw=1.7, label='Empirical Tweedie'),
           Line2D([0], [0], c=C['gray'], ls=(0, (4, 3)), lw=.9,
                  label=r'Empirical benchmark $S_N$')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, 1.003),
           ncol=3, frameon=False, fontsize=8.5, handlelength=2,
           columnspacing=1.7, handletextpad=.5)

LEFT = [.48, 2.58, 4.68]
WIDTH = 1.75
titles = [('(a) Early stopping', 'Tangential accuracy'),
          ('(b) Denoising', 'Normal thickness'),
          ('(c) Total error', 'Joint comparison')]
axs = []
for left, (title, subtitle) in zip(LEFT, titles):
    text(left + WIDTH / 2, 4.46, title, ha='center', va='center',
         fontsize=10, weight='bold')
    text(left + WIDTH / 2, 4.22, subtitle, ha='center', fontsize=8.5)
    ax = axes(left, 2.65, WIDTH, 1.35)
    ax.set_xlim(.40, 0)
    ax.set_xticks([.4, .2, 0], ['0.4', '0.2', '0'])
    ax.tick_params(length=2.5, width=.6, pad=2)
    ax.grid(axis='y', color=C['grid'], lw=.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(NullLocator())
    axs.append(ax)
for x in [2.23, 4.33]:
    arrow(x, 4.46, x + .30, 4.46, scale=7)

x = np.r_[d['eta'] / R, 0.]
a, b, c = axs
a.plot(x, np.r_[d['raw_wt'] / SN, 1.], color=C['raw'], lw=1.7)
a.axhline(1, color=C['gray'], ls=(0, (4, 3)), lw=.85)
a.set_ylim(-.04, 1.14)
a.set_yticks([0, .5, 1], ['0', '0.5', '1'])
a.set_ylabel(r'$W_T/S_N$', labelpad=2)
for num, row, offset in [
    ('1', noisy, (-6, 8)), ('2', minimum, (-4, 9)),
    ('3', {'eta': 0., 'raw_wt': SN}, (-9, 5))]:
    xx, yy = row['eta'] / R, row['raw_wt'] / SN
    a.plot(xx, yy, 'o', ms=3.2, color=C['raw'], clip_on=False)
    a.annotate(num, (xx, yy), textcoords='offset points', xytext=offset,
               fontsize=8, weight='bold', color=C['ink'])

# Add B's exact audited normal readout to the original display grid.
nr = {float(eta): [float(raw), float(tw)] for eta, raw, tw in
      zip(n['eta'], n['raw_wn'], n['tweedie_wn'])}
for row in readouts.values():
    if row['eta'] in nr:
        assert np.allclose(nr[row['eta']], [row['raw_wn'], row['tweedie_wn']],
                           rtol=0, atol=1e-12)
    nr[row['eta']] = [row['raw_wn'], row['tweedie_wn']]
ne = np.array(sorted(nr, reverse=True))
nv = np.array([nr[eta] for eta in ne])
for k, key in enumerate(['raw', 'tw']):
    b.plot(ne / R, nv[:, k] / SN, c=C[key], lw=1.6)
b.set_yscale('log')
b.set_ylim(1e-3, 5)
b.set_yticks([1e-3, 1e-2, 1e-1, 1])
b.yaxis.set_minor_locator(NullLocator())
b.set_ylabel(r'$W_N/S_N$', labelpad=1)

for field, key in [('raw_w2', 'raw'), ('tweedie_w2', 'tw')]:
    c.plot(x, np.r_[d[field] / SN, 1.], color=C[key], lw=1.7)
c.axhline(1, color=C['gray'], ls=(0, (4, 3)), lw=.85)
c.set_ylim(0, 4)
c.set_yticks([0, 1, 2, 3, 4])
c.set_ylabel(r'$W_2/S_N$', labelpad=2)

for label, row in readouts.items():
    xx = row['eta'] / R
    for ax, raw_field, tw_field in [(b, 'raw_wn', 'tweedie_wn'),
                                   (c, 'raw_w2', 'tweedie_w2')]:
        raw, tw = row[raw_field] / SN, row[tw_field] / SN
        ax.plot([xx, xx], [raw, tw], c=C['gray'], ls=':', lw=.85)
        ax.plot(xx, raw, 'o', c=C['raw'], ms=3.3, zorder=5)
        ax.plot(xx, tw, 'o', c=C['tw'], ms=3.3, zorder=5)
        ax.annotate(label, (xx, raw if ax is b else max(raw, tw)),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, weight='bold')

text(3.43, 2.30, r'Normalized noise $\eta/R$ (decreasing $\rightarrow$)',
     ha='center', va='center', fontsize=9.5)

for left, title in zip(LEFT, ['Raw trajectory readouts', 'Matched local patches',
                              'Matched whole clouds']):
    text(left + WIDTH / 2, 2.06, title, ha='center', fontsize=8.4)
    fig.add_artist(Line2D([left / SIZE[0], (left + WIDTH) / SIZE[0]],
                         [1.995 / SIZE[1]] * 2, transform=fig.transFigure,
                         c=C['grid'], lw=.75))

angle = np.linspace(0, 2 * np.pi, 500)
def full_cloud(left, bottom, side, points, color, limit, endpoint=False):
    ax = axes(left, bottom, side, side)
    ax.plot(np.cos(angle), np.sin(angle), color=C['support'], lw=.5)
    if endpoint:
        ax.scatter(P[:, 0] / R, P[:, 1] / R, c=C['ink'], s=5, lw=0)
    else:
        ax.scatter(points[:, 0] / R, points[:, 1] / R, c=color,
                   s=.65, alpha=.65, lw=0)
        ax.scatter(P[:, 0] / R, P[:, 1] / R, c=C['ink'],
                   marker='+', s=5, lw=.35)
        assert np.max(abs(points / R)) < limit
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal')
    ax.axis('off')
    return ax

for cy, num, title, eta, key in [
    (1.72, '1', 'Noisier cloud', noisy['eta'], 'inset_noisy'),
    (1.20, '2', 'Tangential minimum', minimum['eta'], 'inset_minimum'),
    (.68, '3', 'Empirical endpoint', 0., 'endpoint')]:
    full_cloud(LEFT[0] + .01, cy - .24, .48, cloud[key], C['raw'],
               limit=2.1, endpoint=(eta == 0))
    text(LEFT[0] + .55, cy + .055, num + '  ' + title, fontsize=7.8,
         weight='bold')
    text(LEFT[0] + .55, cy - .10, rf'$\eta/R={eta/R:.3g}$', fontsize=8)

patch_counts = {}
posterior_differences = {}
for label, key, cy in [('A', 'a', 1.39), ('B', 'b', .53)]:
    row = readouts[label]
    eta = row['eta']
    basis = cloud['local_basis']
    raw_local = cloud['raw_' + key] @ basis.T - np.array([0., R])
    ids = np.flatnonzero((abs(raw_local[:, 0]) < 1.25) &
                         (raw_local[:, 1] > -.7) & (raw_local[:, 1] < .55))
    patch_counts[label] = len(ids)
    if label == 'A':
        assert np.array_equal(ids, cloud['normal_patch_ids'])
    logits = -np.sum((cloud['raw_' + key][:, None] - P[None]) ** 2, axis=2)/(2*eta**2)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    delta = float(np.max(abs(weights @ P - cloud['tweedie_' + key])))
    assert delta < 1e-12
    posterior_differences[label] = delta

    left = LEFT[1]
    text(left + WIDTH / 2, cy + .46,
         label + rf'  $\eta/R={eta/R:.3g}$', ha='center', weight='bold', fontsize=8)
    for offset, source, col, name in [(.01, 'raw_', C['raw'], 'Raw'),
                                     (.98, 'tweedie_', C['tw'], 'Tweedie')]:
        points = cloud[source + key][ids] @ basis.T - np.array([0., R])
        assert points[:, 0].min() > -1.45 and points[:, 0].max() < 1.45
        assert points[:, 1].min() > -.82 and points[:, 1].max() < .65
        ax = axes(left + offset, cy - .14, .76, .39)
        xx = np.linspace(-1.45, 1.45, 300)
        ax.plot(xx, np.sqrt(R*R - xx*xx) - R, c=C['support'], lw=.6)
        ax.scatter(points[:, 0], points[:, 1], c=col, s=1.2, alpha=.7, lw=0)
        train = P @ basis.T - np.array([0., R])
        ax.scatter(train[:, 0], train[:, 1], c=C['ink'], marker='+', s=7, lw=.35)
        ax.set(xlim=(-1.45, 1.45), ylim=(-.82, .65), aspect='equal')
        ax.axis('off')
        text(left + offset + .38, cy + .30, name, color=col,
             ha='center', fontsize=7.6)
    arrow(left + .79, cy + .045, left + .95, cy + .045, scale=6)

    left = LEFT[2]
    title = 'Lower total error' if label == 'A' else 'Higher total error'
    text(left + WIDTH / 2, cy + .46, label + '  ' + title,
         ha='center', weight='bold', fontsize=8)
    for offset, source, col, name in [(.125, 'raw_', C['raw'], 'Raw'),
                                     (1.075, 'tweedie_', C['tw'], 'Tweedie')]:
        full_cloud(left + offset, cy - .26, .52, cloud[source + key],
                   col, limit=1.6)
        text(left + offset + .26, cy + .30, name, ha='center',
             color=col, fontsize=7.6)
    arrow(left + .77, cy, left + .98, cy, scale=6)

text(3.40, .18,
     r'$W_2(p,\nu_{N,\eta})\ \leq\ W_T(p,\mu_{N,\eta})'
     r'\ +\ \delta_T\ +\ W_N(\nu_{N,\eta})$',
     ha='center', va='center', fontsize=10)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside = []
for txt in fig.findobj(Text):
    if not txt.get_visible() or not txt.get_text():
        continue
    bb = txt.get_window_extent(renderer)
    if bb.width and bb.height and (bb.x0 < -1 or bb.y0 < -1 or
                                  bb.x1 > fig.bbox.width + 1 or
                                  bb.y1 > fig.bbox.height + 1):
        outside.append({'text': txt.get_text(), 'bounds': list(bb.bounds)})
assert not outside, outside

for fmt in ['pdf', 'svg', 'png']:
    (OUT / fmt).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fmt / ('figure_section5_unified.' + fmt), dpi=360)
plt.close(fig)

audit = {
    'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in DATA.iterdir() if p.is_file()},
    'new_simulation': False, 'new_transport_estimation': False,
    'coordinate': 'eta/R, linear and decreasing, shared limits [0.4, 0]',
    'normalization': 'All plotted errors divided by the same S_N',
    'empirical_endpoint': SN, 'radius': R, 'training_count': len(P),
    'particle_count': int(len(cloud['raw_a'])),
    'readouts': readouts, 'tangential_grid_minimum': minimum,
    'normal_plot_eta_range': [float(ne.min()), float(ne.max())],
    'patch_counts': patch_counts,
    'patch_selection': 'Fixed spatial input window at each noise; identical IDs before/after',
    'posterior_pair_max_differences': posterior_differences,
    'same_spatial_scale_within_each_detail_column': True,
    'outside_canvas_text': outside, 'figure_size_inches': list(SIZE),
}
(OUT / 'figure_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print('Unified figure saved. Frozen data, exact readouts, paired particles and canvas checks passed.')
