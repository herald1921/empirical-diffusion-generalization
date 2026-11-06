# Figure 7: paired raw and denoised errors

This additive 2026-11-06 release draws raw W2/WT/WN, denoised W2/WN,
and the tangential diversity deficit G(r). The original beta0/betak curves,
computed Betti-agreement bands, sixteen-site designs, noise grid, geometry
snapshots, and empirical normalization are retained. G is omitted for the
self-intersecting figure eight. Its ellipse/sphere domains and the segment
extension follow the 2026-11-05 diversity release.

The old denoised WT curve is no longer displayed. Historical code is untouched.
Five distributional errors plus G give six curves (five for the figure eight).
Zero values are not plotted on logarithmic axes; the segment's denoised WN is
identified explicitly as zero.

## Reproduction

Dependencies: Python, NumPy, SciPy, Matplotlib, PyMuPDF.
From the repository root:

```sh
python appendix_c_geometries/synthetic_six_curves_20261106/compute_curves.py --repository .
python appendix_c_geometries/synthetic_six_curves_20261106/plot_six_curves.py --repository .
```

The scripts import the archived `figure8_appendixC_20261019/generate_figure8.py`,
its `output/figure8_audit.json`, and its original design/metadata dependencies.
The renderer also reads `synthetic_diversity_20261105/diversity_curves.csv` and
`inputs/synthetic_geometry_template.pdf`. Keep those directories in sparse
checkouts. Only geometry snapshots are copied from the template; old axis
labels are removed from the in-memory copy before placement.

## Paired computation and checks

At all 16 original noise values, raw and denoised diagnostics use the same
512 equally weighted Gaussian-mixture nodes, 512 population nodes, and
resolution-matched empirical endpoint. Squared Euclidean optimal transport
uses SciPy linear assignment. Normal errors use nearest-point distances.
The raw curves are newly computed, not inferred from the denoised curves.
Retained denoised W2/WN and empirical endpoints reproduce the archived values
to absolute tolerance 1e-12. Finite-node tangential/normal bounds are checked.

`numeric_audit.json` includes 1024-node refinement and a second 512-node scramble
at eta=0.06, 0.38, 1.4. These are numerical sensitivity checks, not certified
continuum errors or guarantees at untested scales. `curves.json` stores the
actual five error values; all G and Betti inputs retain their original records.

## Appearance

STIXGeneral, 9.2 pt ordinary text, with the Figure 2 palette:
raw W2 blue #0072B2; raw WT green #009E73; raw WN gold #E69F00 dashed;
denoised W2 vermilion #D55E00 dash-dot; denoised WN gold solid;
G pink #CC79A7 dotted. Betti curves remain gray-blue dashed/dotted.
The six error/deficit labels share one row; Betti and benchmark labels share
a second row. Row spacing is compacted without scaling the glyphs or snapshots.
Outputs: `output/figure_07_synthetic_geometries.pdf`, its PNG preview,
and `output/plot_audit.json`.
