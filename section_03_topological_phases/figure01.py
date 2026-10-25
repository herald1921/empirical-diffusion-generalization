"""Figure 1: three topology regimes and matched empirical diffusion paths.

Reproduce: python make_figure_01.py
Required: numpy, scipy, matplotlib. Optional independent audit: gudhi.
The script runs a deterministic illustration, saves all numerical data and an
integration audit, and exports editable vector PDF/SVG plus a 360 dpi PNG.
No neural network, learned score, image-generation model, or hand-drawn paths.
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent
INK = '#253340'
MUTED = '#62717D'
LIGHT = '#C7CED3'
UNION = '#DCEEE9'
UNION_EDGE = '#6AA493'
# Okabe-Ito colors; each path keeps its terminal angular-sector color.
COLORS = np.array(['#009E73', '#0072B2', '#CC79A7', '#E69F00'])


@dataclass
class Config:
    n_train: int = 16
    n_particles: int = 512
    n_traces: int = 16
    radius: float = 2.0
    angular_amplitude: float = 0.1
    delta: float = 0.05
    dimension: int = 2
    seed: int = 20260920
    eta_start: float = 2.4
    eta_high: float = 1.2
    eta_middle: float = 0.16
    eta_low: float = 0.008
    n_time: int = 800
    rtol: float = 2e-9
    atol: float = 2e-11
    max_step: float = 0.055


def training_design(cfg):
    """Deterministic midpoint quantiles of (1+a cos(theta))/(2 pi)."""
    u = (np.arange(cfg.n_train) + 0.5) / cfg.n_train
    theta = 2 * np.pi * u
    for _ in range(12):
        theta -= (theta + cfg.angular_amplitude * np.sin(theta) - 2*np.pi*u) / (
            1 + cfg.angular_amplitude * np.cos(theta))
    points = cfg.radius * np.column_stack([np.cos(theta), np.sin(theta)])
    return theta, points


def geometry(cfg, theta):
    gaps = np.diff(np.r_[theta, theta[0] + 2*np.pi])
    if np.max(gaps) >= np.pi:
        raise ValueError('The exact circle-union formula needs all gaps < pi.')
    bridges = cfg.radius * np.sin(gaps / 2)
    c = np.sqrt(cfg.dimension) + np.sqrt(2*np.log(1/cfg.delta))
    h = 2 * cfg.radius * np.sin(np.max(gaps)/4)
    # paper_v3/Sections/Topology.tex: eq:top-reconstruction-radii
    # and cor:top-mass-window. This is the general reach-based theorem,
    # distinct from the sharper circle-specific condition h < c*eta < R.
    rho = cfg.radius
    if not 0 < h < (2-np.sqrt(2))*rho:
        raise ValueError('The reconstruction theorem requires h < (2-sqrt(2))*rho.')
    dh = np.sqrt(4*rho**4 - 12*rho*rho*h*h + h**4)
    aminus = (2*rho*rho-h*h-dh)/(4*rho)
    aplus = (2*rho*rho-h*h+dh)/(4*rho)
    rminus = np.sqrt(2*rho*aminus-aminus*aminus)
    rplus = np.sqrt(2*rho*aplus-aplus*aplus)
    return dict(theory_reach=float(rho), theory_r_minus=float(rminus),
                theory_r_plus=float(rplus), theory_eta_lower=float(rminus/c),
                theory_eta_upper=float(rplus/c),
                c=float(c), covering_radius=float(h),
                connectivity_radius=float(bridges.max()),
                disjoint_radius=float(bridges.min()),
                eta_hole_birth=float(cfg.radius/c),
                eta_fragmentation=float(bridges.max()/c),
                eta_all_disjoint=float(bridges.min()/c),
                eta_sufficient_lower=float(h/c)), bridges


def union_betti(radius, cfg, bridges):
    """Exact Betti numbers for equal-radius disks centered on this circle.

    Below R, each unbridged consecutive angular gap cuts one component;
    once every gap is bridged the union is an annulus. At r >= R every
    disk contains the origin, so their union is star-shaped.
    """
    if radius >= cfg.radius:
        return 1, 0
    open_gaps = int(np.count_nonzero(bridges > radius))
    return max(1, open_gaps), int(open_gaps == 0)


def eta_to_time(eta):
    """Cosine VP reverse time: alpha=sin(pi*t/2), sigma=cos(pi*t/2).

    The displayed state is U=X/alpha, so the target circle is fixed.
    Noise-to-data time increases, eta=cot(pi*t/2) decreases.
    """
    return 2/np.pi*np.arctan2(1., np.asarray(eta))


def normalized_velocity(x, eta, points):
    t = float(eta_to_time(eta))
    return np.pi/np.sin(np.pi*t) * (posterior_mean(x, eta, points)-x)


def posterior_mean(x, eta, points):
    d2 = np.sum((x[:, None, :] - points[None, :, :])**2, axis=2)
    return softmax(-d2 / (2*eta*eta), axis=1) @ points


def simulate(cfg, points):
    rng = np.random.default_rng(cfg.seed)
    source_atoms = rng.integers(cfg.n_train, size=cfg.n_particles)
    z = rng.standard_normal((cfg.n_particles, 2))
    initial = points[source_atoms] + cfg.eta_start * z
    snapshots = np.array([cfg.eta_high, cfg.eta_middle, cfg.eta_low])
    eta = np.unique(np.r_[np.geomspace(cfg.eta_start, cfg.eta_low, cfg.n_time), snapshots])[::-1]
    s = np.log(cfg.eta_start / eta)

    def rhs(log_noise, flat):
        x = flat.reshape(-1, 2)
        noise = cfg.eta_start * np.exp(-log_noise)
        return (posterior_mean(x, noise, points) - x).ravel()

    def integrate(rtol, atol, step):
        sol = solve_ivp(rhs, (0., float(s[-1])), initial.ravel(),
                        method='DOP853', t_eval=s, rtol=rtol, atol=atol,
                        max_step=step)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y.T.reshape(-1, cfg.n_particles, 2), sol.nfev

    paths, nfev = integrate(cfg.rtol, cfg.atol, cfg.max_step)
    checked, nfev_check = integrate(cfg.rtol/5, cfg.atol/5, cfg.max_step/2)
    refinement = np.linalg.norm(paths-checked, axis=2)
    if refinement.max() > 2e-5:
        raise RuntimeError(f'Integration refinement error: {refinement.max():.3g}')
    # Use the tighter integration in the published figure and saved arrays.
    paths = checked
    snap_indices = np.array([np.argmin(abs(eta-e)) for e in snapshots])
    end_angles = np.mod(np.arctan2(paths[-1, :, 1], paths[-1, :, 0]), 2*np.pi)
    sectors = np.floor(end_angles/(np.pi/2)).astype(int).clip(0, 3)
    # Display-only angular stratification; all particles appear in the clouds.
    # No trajectory is selected according to speed, shape, or an error metric.
    selected = []
    for target in 2*np.pi*(np.arange(cfg.n_traces)+0.25)/cfg.n_traces:
        distance = abs(np.angle(np.exp(1j*(end_angles-target))))
        distance[selected] = np.inf
        selected.append(int(np.argmin(distance)))
    audit = dict(solver='DOP853 in s=log(eta_start/eta)',
                 nfev=nfev, nfev_refinement=nfev_check,
                 max_path_refinement_error=float(refinement.max()),
                 terminal_max_refinement_error=float(refinement[-1].max()),
                 snapshot_indices=snap_indices.tolist(),
                 visible_trajectory_ids=selected,
                 initial_law='p_N convolved with N(0, eta_start^2 I_2)',
                 cloud_sampling=f'{cfg.n_particles} iid initial particles, then one deterministic ODE coupling',
                 display_selection=f'closest terminal angle to each of {cfg.n_traces} equally spaced directions')
    return eta, paths, snapshots, snap_indices, sectors, np.array(selected), source_atoms, z, audit


def reverse_time_audit(cfg, points, eta, paths, selected):
    """Independently integrate the same highlighted paths in reverse time t."""
    times = eta_to_time(eta)
    initial = paths[0, selected].ravel()
    def rhs(t, flat):
        noise = 1/np.tan(np.pi*t/2)
        return normalized_velocity(flat.reshape(-1,2),noise,points).ravel()
    sol = solve_ivp(rhs,(float(times[0]),float(times[-1])),initial,
                    t_eval=times,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.002)
    if not sol.success:
        raise RuntimeError(sol.message)
    direct = sol.y.T.reshape(-1,len(selected),2)
    error = float(np.linalg.norm(direct-paths[:,selected],axis=2).max())
    if error > 2e-5:
        raise RuntimeError(f'Reverse-time reparameterization failed: {error}')
    return dict(direct_t_integration_max_error=error,checked_paths=len(selected))


def topology_audit(cfg, points, geom, bridges):
    noises = np.geomspace(cfg.eta_start, cfg.eta_low, 401)
    exact = np.array([union_betti(geom['c']*e, cfg, bridges) for e in noises])
    result = dict(method='exact circular-gap formula', checked_scales=len(noises))
    try:
        import gudhi
    except ImportError:
        result['independent_alpha_complex_check'] = 'not run: optional gudhi is not installed'
        return result
    st = gudhi.AlphaComplex(points=points).create_simplex_tree()
    st.compute_persistence(homology_coeff_field=2)
    intervals = [st.persistence_intervals_in_dimension(d) for d in (0, 1)]
    # Remove floating-point zero-persistence features at the co-circular center.
    intervals = [bars[(bars[:, 1]-bars[:, 0]) > 1e-10] for bars in intervals]
    numerical = np.array([[np.count_nonzero((b[:, 0] <= (geom['c']*e)**2) &
                                            ((geom['c']*e)**2 < b[:, 1]))
                           for b in intervals] for e in noises])
    if not np.array_equal(exact, numerical):
        raise RuntimeError('Circle formula does not agree with alpha-complex persistence.')
    result.update(independent_alpha_complex_check='passed', gudhi_version=gudhi.__version__,
                  coefficient_field=2, zero_persistence_tolerance_squared_radius=1e-10)
    return result


def setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'mathtext.fontset': 'dejavusans', 'axes.labelsize': 7.3,
        'axes.titlesize': 8.5, 'axes.edgecolor': LIGHT,
        'axes.linewidth': .65, 'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.0, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
        'text.color': INK, 'axes.labelcolor': MUTED,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'path.simplify': False,
    })


def section(fig, y, letter, text, suffix=None):
    fig.text(.062, y, letter, weight='bold', size=9.2, va='center')
    fig.text(.089, y, text, weight='bold', size=8.5, va='center')
    if suffix:
        fig.text(.966, y, suffix, ha='right', va='center', size=7.2, color=MUTED)


def frame(ax, lim, cfg, ticks):
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(pad=1.8)
    for spine in ax.spines.values():
        spine.set_color('#D3DADD')
    ax.set_facecolor('white')


def target_and_samples(ax, points, cfg, local=False, size=6.0):
    if local:
        yy = np.linspace(-.55, .55, 300)
        ax.plot(yy/cfg.radius, (np.sqrt(cfg.radius**2-yy**2)-cfg.radius)/cfg.radius,
                color=INK, lw=.85, ls=(0, (3, 2)), zorder=4)
        visible = (points[:, 0] > 1.7) & (abs(points[:, 1]) < .56)
        ax.scatter(points[visible, 1]/cfg.radius,
                   (points[visible, 0]-cfg.radius)/cfg.radius,
                   s=size, c=INK, linewidths=.5, edgecolors='white', zorder=9)
    else:
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=INK, lw=.75,
                            ls=(0, (3, 2)), zorder=4))
        ax.scatter(points[:, 0]/cfg.radius, points[:, 1]/cfg.radius,
                   s=size, c=INK, linewidths=.35, edgecolors='white', zorder=9)


def draw_union(ax, noise, points, cfg, geom, bridges, column):
    radius = geom['c'] * noise
    lim = [3.5, 1.43, 1.43][column]
    frame(ax, lim, cfg, [-3, 0, 3] if column == 0 else [-1, 0, 1])
    # Opaque equal-color disks depict the union without multiplicity shading.
    for p in points/cfg.radius:
        ax.add_patch(Circle(p, radius/cfg.radius, facecolor=UNION,
                            edgecolor='none', zorder=1))
    # A distance-field outline shows only the boundary of the union.
    grid = np.linspace(-lim, lim, 650)
    xx, yy = np.meshgrid(grid, grid)
    nearest = np.full_like(xx, np.inf)
    for p in points/cfg.radius:
        np.minimum(nearest, (xx-p[0])**2+(yy-p[1])**2, out=nearest)
    ax.contour(xx, yy, nearest, levels=[(radius/cfg.radius)**2],
               colors=[UNION_EDGE], linewidths=.55, zorder=2)
    ax.scatter(points[:, 0]/cfg.radius, points[:, 1]/cfg.radius,
               s=1.9 if column < 2 else .45, c=INK, edgecolors='none', zorder=9)
    if column > 0:
        ax.add_patch(Rectangle((1.75/cfg.radius, -.52/cfg.radius),
                               .48/cfg.radius, 1.04/cfg.radius, fill=False,
                               ec='#6C7D87', lw=.6, ls=(0, (2.5, 2)), zorder=10))
    b0, b1 = union_betti(radius, cfg, bridges)
    ax.text(.5, -.19, rf'$r={radius:.3f}\quad (\beta_0,\beta_1)=({b0},{b1})$',
            transform=ax.transAxes, ha='center', va='center', fontsize=7.6)
    if column == 0:
        ax.set_ylabel('$u_2/R$', labelpad=1)


def draw_union_inset(fig, center_x, center_y, noise, points, cfg, geom):
    """Magnify the same right-hand arc so the late disks remain distinguishable."""
    ax = fig.add_axes([center_x-.066, center_y-.027, .132, .054])
    ax.set_xlim(-.26, .26)
    ax.set_ylim(-.125, .115)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    radius = geom['c']*noise/cfg.radius
    xy = np.column_stack([points[:, 1], points[:, 0]-cfg.radius])/cfg.radius
    for p in xy:
        ax.add_patch(Circle(p, radius, facecolor=UNION, edgecolor='none'))
    # Draw the union outline, not the boundaries of overlapping disks.
    gx, gy = np.meshgrid(np.linspace(-.26, .26, 450), np.linspace(-.125, .115, 210))
    near = np.full_like(gx, np.inf)
    for p in xy:
        np.minimum(near, (gx-p[0])**2+(gy-p[1])**2, out=near)
    if near.min() < radius*radius < near.max():
        ax.contour(gx, gy, near, levels=[radius*radius], colors=[UNION_EDGE], linewidths=.6)
    ax.scatter(xy[:, 0], xy[:, 1], s=3.2, color=INK, zorder=3)
    ax.text(.5, 1.12, 'local detail', transform=ax.transAxes, ha='center',
            va='bottom', color=MUTED, size=6.5)
    for spine in ax.spines.values():
        spine.set_color('#ADBABF')
        spine.set_linewidth(.6)


def path_arrow(ax, xy, color, local=False):
    # Locate the arrow by path arclength, independently of visual appearance.
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if step.sum() < .014:
        return
    cum = np.r_[0., np.cumsum(step)]
    j = max(2, min(len(xy)-1, np.searchsorted(cum, .65*cum[-1])))
    j0 = max(0, j-max(2, len(xy)//20))
    ax.annotate('', xy=xy[j], xytext=xy[j0],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=.65,
                                mutation_scale=4.6, shrinkA=0, shrinkB=0),
                annotation_clip=True, zorder=7)


def draw_vector_field(ax, noise, points, cfg, column, local=False):
    """True normalized PF direction; normalize glyph lengths for readability.

    Grey arrows encode direction only, not the magnitude of the velocity.
    In particular they are not manually forced to point towards the circle.
    """
    if local:
        xx, yy = np.meshgrid(np.linspace(-.24,.24,9), np.linspace(-.105,.095,5))
        query = np.column_stack([cfg.radius*(1+yy.ravel()), cfg.radius*xx.ravel()])
        position = np.column_stack([xx.ravel(), yy.ravel()])
        velocity = normalized_velocity(query, noise, points)[:, ::-1]/cfg.radius
        length = .023
    elif column == 0:
        xx, yy = np.meshgrid(np.linspace(-3.1,3.1,11), np.linspace(-3.1,3.1,11))
        position = np.column_stack([xx.ravel(),yy.ravel()])
        velocity = normalized_velocity(position*cfg.radius,noise,points)/cfg.radius
        length = .25
    else:
        theta = np.linspace(0,2*np.pi,32,endpoint=False)+.021
        radii = [.76,.89,1.11,1.26] if column == 1 else [.93,1.,1.07]
        position = np.concatenate([r*np.column_stack([np.cos(theta),np.sin(theta)]) for r in radii])
        velocity = normalized_velocity(position*cfg.radius,noise,points)/cfg.radius
        length = .115 if column == 1 else .075
    speed = np.linalg.norm(velocity,axis=1)
    valid = speed > 1e-9
    unit = velocity[valid]/speed[valid,None]*length
    ax.quiver(position[valid,0],position[valid,1],unit[:,0],unit[:,1],
              angles='xy',scale_units='xy',scale=1,pivot='mid',
              color='#B6BEC6',alpha=.88,width=.0029,
              headwidth=3.4,headlength=4.4,headaxislength=3.8,zorder=1.5)


def draw_raw(ax, paths, start, stop, points, cfg, sectors, trace_ids, column, noise, local=False):
    current = paths[stop]
    if local:
        ax.set_xlim(-.26, .26)
        ax.set_ylim(-.125, .115)
        ax.set_aspect('equal')
        ax.set_xticks([-.2, 0, .2])
        ax.set_yticks([-.1, 0, .1])
        ax.tick_params(pad=1.6)
        ax.set_xlabel('$u_2/R$', labelpad=1)
        ax.set_ylabel('$(u_1-R)/R$', labelpad=1)
        transform = lambda a: np.stack([a[..., 1], a[..., 0]-cfg.radius], axis=-1)/cfg.radius
        # Include every local terminal particle; other arcs are never folded in.
        ids = np.flatnonzero((current[:, 0] > 1.7) & (abs(current[:, 1]) < .63))
        for i in ids:
            seg = transform(paths[start:stop+1, i])
            ax.plot(seg[:, 0], seg[:, 1], color=COLORS[sectors[i]],
                    alpha=.18, lw=.45, zorder=2)
        cloud_size, cloud_alpha = 5., .60
    else:
        frame(ax, [3.5, 1.43, 1.43][column], cfg,
              [-3, 0, 3] if column == 0 else [-1, 0, 1])
        transform = lambda a: a/cfg.radius
        ids = np.arange(cfg.n_particles)
        cloud_size, cloud_alpha = 3.3, .34
        if column == 0:
            ax.set_ylabel('$u_2/R$', labelpad=1)
        ax.set_xlabel('$u_1/R$', labelpad=1)
    draw_vector_field(ax, noise, points, cfg, column, local=local)
    cur = transform(current[ids])
    ax.scatter(cur[:, 0], cur[:, 1], s=cloud_size,
               c=COLORS[sectors[ids]], alpha=cloud_alpha, edgecolors='none', zorder=3)
    for i in trace_ids:
        if local and i not in ids:
            continue
        seg = transform(paths[start:stop+1, i])
        color = COLORS[sectors[i]]
        ax.plot(seg[:, 0], seg[:, 1], color='white', lw=1.85 if not local else 1.5,
                alpha=.9, solid_capstyle='round', zorder=5)
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=.9 if not local else .8,
                alpha=.95, solid_capstyle='round', zorder=6)
        path_arrow(ax, seg, color, local)
        ax.scatter(*seg[-1], s=10 if not local else 13, color=color,
                   edgecolors='white', linewidths=.4, zorder=8)
    target_and_samples(ax, points, cfg, local=local, size=6 if not local else 12)
    if not local and column > 0:
        ax.add_patch(Rectangle((1.75/cfg.radius, -.52/cfg.radius),
                               .48/cfg.radius, 1.04/cfg.radius, fill=False,
                               ec='#6C7D87', lw=.7, ls=(0, (2.5, 2.0)), zorder=10))


def draw_timeline(fig, cfg, geom, bridges, snapshots):
    thresholds = np.r_[cfg.radius, bridges]/geom['c']
    noises = np.unique(np.r_[np.geomspace(1e6, 1e-6, 1200),
                            thresholds*(1-1e-8), thresholds*(1+1e-8)])[::-1]
    times = eta_to_time(noises)
    b = np.array([union_betti(geom['c']*e, cfg, bridges) for e in noises])
    ax0 = fig.add_axes([.092, .096, .855, .045])
    ax1 = fig.add_axes([.092, .049, .855, .032], sharex=ax0)
    t_left, t_right = eta_to_time([geom['theory_eta_upper'],geom['theory_eta_lower']])
    for ax in (ax0, ax1):
        ax.set_xlim(0,1)
        ax.axvspan(t_left,t_right,color='#EFF0F2',lw=0,zorder=0)
        for boundary in (t_left,t_right):
            ax.axvline(boundary,color='#AAB0B6',lw=.75,ls=(0,(3,3)),zorder=1)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(which='minor',bottom=False)
    ax0.plot(times,b[:,0],color='#0072B2',lw=1.1,zorder=3)
    ax1.plot(times,b[:,1],color='#008F70',lw=1.1,zorder=3)
    ax0.set_ylim(-2,cfg.n_train*1.1)
    ax0.set_yticks([1,cfg.n_train])
    ax0.set_ylabel(r'$\beta_0$',rotation=0,labelpad=7,va='center',color=INK)
    ax0.tick_params(axis='x',labelbottom=False,bottom=False)
    ax1.set_ylim(-.13,1.18)
    ax1.set_yticks([0,1])
    ax1.set_ylabel(r'$\beta_1$',rotation=0,labelpad=7,va='center',color=INK)
    ax1.set_xticks([0,.2,.4,.6,.8,1.],['0','0.2','0.4','0.6','0.8','1'])
    fig.text(.092,.010,r'Reverse diffusion time $t$',color=MUTED,size=7)
    fig.text(.95,.010,r'Noise $\longrightarrow$ data',ha='right',color=MUTED,size=7)
    for t,label in zip(eta_to_time(snapshots),['I','II','III']):
        ax0.text(t,cfg.n_train*1.25,label,ha='center',va='bottom',size=7.2,weight='bold')
    ax0.text((t_left+t_right)/2,cfg.n_train*.70,
             'Topology recovery interval'+chr(10)+rf'${t_left:.3f}<t<{t_right:.3f}$',
             ha='center',va='center',fontsize=6.5,linespacing=1.05,color='#777F86')


def make_figure(cfg, theta, points, geom, bridges, eta, paths, snapshots, snap_indices, sectors, trace_ids, out):
    setup_style()
    fig = plt.figure(figsize=(6.75, 7.65))
    columns = [.075, .383, .691]
    width = .257
    titles = [('Phase I: Linear regime', 'Mean averaging'),
              ('Phase II: Geometric regime', 'Manifold reconstruction'),
              ('Phase III: Overfitting regime', 'Nearest-neighbor collapse')]
    section(fig, .982, 'a', 'High-probability neighborhoods')
    for j, (title, subtitle) in enumerate(titles):
        x = columns[j] + width/2
        fig.text(x, .954, title, ha='center', weight='bold', size=8.0)
        fig.text(x, .933, subtitle, ha='center', size=7.6, color=INK)
        fig.text(x, .913, rf'$t={eta_to_time(snapshots[j]):.3f}$  $(\eta={snapshots[j]:g})$', ha='center', color=MUTED, size=7.4)
        ax = fig.add_axes([columns[j], .677, width, .225])
        draw_union(ax, snapshots[j], points, cfg, geom, bridges, j)
    for j in (1, 2):
        draw_union_inset(fig, columns[j]+width/2, .787, snapshots[j], points, cfg, geom)
    section(fig, .613, 'b', 'Probability-flow trajectories')
    starts = np.r_[0, snap_indices[:-1]]
    for j in range(3):
        ax = fig.add_axes([columns[j], .363, width, .215])
        movement = ['Mean-directed flow', 'Normal contraction', 'Tangential fitting'][j]
        fig.text(columns[j]+width/2,.590,movement,ha='center',size=8,color=MUTED)
        draw_raw(ax, paths, starts[j], snap_indices[j], points, cfg, sectors, trace_ids, j, snapshots[j])
    section(fig, .303, 'c', 'Local trajectory geometry')
    for j in [1, 2]:
        ax = fig.add_axes([columns[j], .211, width, .08])
        draw_raw(ax, paths, starts[j], snap_indices[j], points, cfg, sectors, trace_ids, j, snapshots[j], local=True)
    # A compact visual key uses the remaining cell; no redundant method legend.
    key = fig.add_axes([columns[0]-.005, .204, width+.015, .081])
    key.set_axis_off()
    handles = [Line2D([], [], color=INK, lw=.9, ls=(0, (3, 2))),
               Line2D([], [], color=INK, marker='o', lw=0, ms=3.3),
               Line2D([], [], color=COLORS[1], marker='o', lw=.9, ms=3.3)]
    key.legend(handles, ['Target circle', 'Training samples', 'Paths and current points'],
               frameon=False, loc='upper left', borderaxespad=0,
               handlelength=2, handletextpad=.7, labelspacing=.65, fontsize=7.0)
    fig.text(columns[0]-.005, .199, 'Grey arrows: flow direction only', size=6.5, color=MUTED)
    section(fig, .165, 'd', 'Topological phase characterization')
    draw_timeline(fig, cfg, geom, bridges, snapshots)
    stem = out/'figure_01_topology_trajectories_n16_v4'
    fig.savefig(stem.with_suffix('.pdf'), metadata={
        'Title': 'Topology regimes and matched empirical diffusion trajectories',
        'Author': '', 'Subject': 'Circle illustration; exact empirical score; paired particles',
        'Keywords': 'diffusion, topology, circle, probability flow, JMLR'})
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.png'), dpi=360)
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--particles', type=int, default=512)
    parser.add_argument('--traces', type=int, default=16)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output')
    args = parser.parse_args()
    cfg = Config(seed=args.seed, n_particles=args.particles, n_traces=args.traces)
    if not 1 <= cfg.n_traces <= cfg.n_particles:
        parser.error('Need 1 <= traces <= particles.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT/'data'
    data_dir.mkdir(exist_ok=True)
    theta, points = training_design(cfg)
    geom, bridges = geometry(cfg, theta)
    topo_check = topology_audit(cfg, points, geom, bridges)
    eta, paths, snapshots, snap_indices, sectors, trace_ids, source_atoms, z, audit = simulate(cfg, points)
    audit.update(reverse_time_audit(cfg,points,eta,paths,trace_ids))
    for e, expected in zip(snapshots, [(1, 0), (1, 1), (cfg.n_train, 0)]):
        if union_betti(geom['c']*e, cfg, bridges) != expected:
            raise RuntimeError('The selected snapshots do not represent the stated regimes.')
    if not geom['theory_eta_lower'] < cfg.eta_middle < geom['theory_eta_upper']:
        raise RuntimeError('Middle snapshot is outside the reconstruction theorem window.')
    # Every displayed current particle must be visible in the full panels.
    snapshot_lims = np.array([3.5, 1.43, 1.43])*cfg.radius
    outside = [int(np.any(abs(paths[i]) > bound, axis=1).sum())
               for i, bound in zip(snap_indices, snapshot_lims)]
    if any(outside):
        raise RuntimeError(f'Current particles outside the displayed frames: {outside}')
    audit.update(current_particles_outside_frames=outside,
                 max_abs_current_coordinates=[float(abs(paths[i]).max()) for i in snap_indices])
    np.savez_compressed(data_dir/'figure_01_data.npz', theta=theta, training_points=points,
                        eta=eta, reverse_time=eta_to_time(eta), raw_paths=paths, snapshots=snapshots,
                        snapshot_times=eta_to_time(snapshots),
                        snapshot_indices=snap_indices, color_sectors=sectors,
                        highlighted_ids=trace_ids, source_atoms=source_atoms,
                        initial_standard_normal=z)
    report = dict(config=asdict(cfg), geometry=geom,
                  time_convention=dict(schedule='cosine VP, noise-to-data',
                      alpha='sin(pi*t/2)',sigma='cos(pi*t/2)',eta='cot(pi*t/2)',
                      coordinates='U=X/alpha, signal-normalized',
                      normalized_velocity='pi/sin(pi*t) * (m_eta(U)-U)',
                      initial_time=float(eta_to_time(cfg.eta_start)),
                      theory_time_interval=eta_to_time([geom['theory_eta_upper'],geom['theory_eta_lower']]).tolist()),
                  snapshots=[dict(t=float(eta_to_time(e)),eta=float(e), ball_radius=float(geom['c']*e),
                                  betti=list(union_betti(geom['c']*e, cfg, bridges))) for e in snapshots],
                  numerical_audit=audit, topology_audit=topo_check,
                  software=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, matplotlib=matplotlib.__version__))
    (data_dir/'figure_01_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    stem = make_figure(cfg, theta, points, geom, bridges, eta, paths, snapshots,
                       snap_indices, sectors, trace_ids, out)
    print(json.dumps(dict(output=str(stem), snapshots=report['snapshots'],
                          audit=audit, topology_audit=topo_check), indent=2))


if __name__ == '__main__':
    main()
