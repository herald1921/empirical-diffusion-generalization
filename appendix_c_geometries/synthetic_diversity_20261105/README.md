# Synthetic geometries with tangential diversity deficit

This additive release preserves the frozen synthetic designs, geometry panels,
transport curves, topology calculations, and empirical normalization of the
2026-10-19 experiment. It adds the manuscript's projected-neighborhood deficit
`G(r) = inf_{supp(v) subset Pi(U_r)} W2(p,v)` to ellipse, sphere, and segment
panels. No diversity curve is assigned to the self-intersecting figure eight.
Historical experiment directories remain unchanged.

## Reproduce

Python dependencies: NumPy, SciPy, Matplotlib, and PyMuPDF (`fitz`).
Run the following from the repository root:

```sh
python appendix_c_geometries/synthetic_diversity_20261105/compute_diversity.py --repository .
python appendix_c_geometries/synthetic_diversity_20261105/plot_synthetic_diversity.py --repository . --input appendix_c_geometries/synthetic_diversity_20261105/inputs/synthetic_geometry_template.pdf
```

The calculation also runs independently without `--repository`, using bundled
frozen input arrays. Supplying that argument checks their original hashes and
the original empirical baseline. Default sphere integration uses 2^20 scrambled
Sobol points; `--sphere-power` changes that exponent.

The renderer imports the unchanged historical
`appendix_c_geometries/figure8_appendixC_20261019/generate_figure8.py`, reads its
`output/figure8_audit.json`, and reads the designs and metadata under
`appendix_c_geometries/appendix_jmlr_revision_20261018/evidence_bundle`.
These original files are required when rendering, including in a sparse checkout.
The supplied template PDF is used only to recover the original synthetic geometry
vector panels; new text and curves are rendered separately.

## Data and mathematical scope

`diversity_curves.csv` contains `shape,eta,r,G,G_over_SN,defined,exact_zero`.
There are 422 noise values per included shape, including a zero-noise diagnostic,
the original grid, and a dense grid from 0.008 to 2. Radius is
`r = (sqrt(D) + sqrt(2 log(20))) eta`, with D=2 for ellipse/segment and D=3 for sphere.

Ellipse uses axes (2,1), angle density `(1+0.1 cos(theta))/(2 pi)`, and the frozen
16-site design. Its curve is defined only for `r < 0.5`. Sphere uses radius 2,
uniform surface density, and the frozen 16 sites, with `r < 2`. The segment is
uniform on `[-sqrt(3),sqrt(3)]` with 16 equal-cell midpoints; projection is unique
at every radius. Values outside the ellipse/sphere domains are NaN. Exact zeros
remain zero in the data and are omitted on the logarithmic display, without a
positive display floor. There are no figure-eight rows.

The frozen 512-point assignment estimates `S_N` are retained. In particular,
`G(0)` is not forced to equal that equal-weight empirical benchmark. The segment's
continuum value `G(0)=1/16` is slightly above its numerical `S_N` because of baseline
discretization; this is documented in the audit and does not alter the baseline.

## Validation and outputs

`diversity_method.md` gives the geometric reductions and verification procedure.
`diversity_audit.json` records hashes, domains, quadrature checks, and tolerances.
Ellipse independent midpoint integration differs by at most 1.29e-10 in G squared
on the checked radii; independent ellipse projection checks locate all tested
boundary projections inside the computed arcs. Sphere independent Sobol scrambles
differ by at most 9.32e-7 in G on the checked radii. This is a sensitivity check,
not a certified deterministic integration error. Segment values use an exact
formula, independently checked by quadrature. All computed curves are nonincreasing.

The final PDF and PNG are in `output/figure_07_synthetic_geometries.*`.
`output/plot_audit.json` records the current input/data hashes, 9.2-point ordinary
font size, preserved geometry/transport curves, and text-boundary checks.
