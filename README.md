# Topology and Distributional Generalization in Empirical Diffusion Models

Research code accompanying the manuscript by **Jinghuan Sun, Jianxing Wang,
and Yaoyu Zhang** (equal contribution; joint first authors). The repository contains empirical diffusion / flow-matching
experiments, geometric diagnostics, neural-field comparisons, and figure scripts.

**Code:** [https://github.com/herald1921/empirical-diffusion-generalization](https://github.com/herald1921/empirical-diffusion-generalization)  
**License:** [MIT](LICENSE). Dataset and pretrained-model terms are separate.

This code release includes synthetic inputs and numerical records. Obtain image
datasets independently; image-bearing experiment archives and model weights are
not distributed here.

## Compact Figure 2 layout (2026-11-05)

Run `python run.py figure2-uniform` for the current 6 x 4.35 inch layout:
[`section_04_distribution/merged_uniform_20261105/`](section_04_distribution/merged_uniform_20261105/).
It preserves the previous data, colors and 9.2 pt STIX font, while reducing
lower-left whitespace and the height of the right-column plots.
The old layout remains available with `python run.py figure2-uniform-20261104`.

## Revised merged Figure 2 (2026-11-04)

The reorganized Section 4 uses a uniform-circle target with 16 equally spaced
sites, merged stopping/denoising readouts, and ambient-dimension comparisons.
Run `python run.py figure2-uniform-20261104`; see
[`section_04_distribution/merged_uniform_20261104/`](section_04_distribution/merged_uniform_20261104/).
This version is separate from the historical paper map below. All earlier code,
inputs, and figure jobs are retained.

## Current manuscript figure map (2026-11-05)

The reorganized manuscript has nine figures. Historical directories and job
names below are retained for reproducibility; they do not imply the current
figure number. Figure-file names in the manuscript now match this table.

| Current figure | Content | Code |
|---|---|---|
| 1 | Topological phases | `section_03_topological_phases/` |
| 2 | Merged uniform-circle comparison | `section_04_distribution/merged_uniform_20261105/` |
| 3 | CelebA empirical averaging | `section_06_discussion/figure04/` |
| 4 | Empirical and U-Net comparison | `section_06_discussion/figure05/` |
| 5 | Cross-architecture comparison | `section_06_discussion/figure06/` |
| 6 | Field replacement | `section_06_discussion/figure07/` |
| 7 | Synthetic geometries with diversity deficit | `appendix_c_geometries/synthetic_diversity_20261105/` |
| 8 | MNIST comparison | `appendix_d_images/figure09/` |
| 9 | Extended MLP training | `appendix_d_images/figure10/` |

Run `python run.py geometry-diversity` to recompute the added deficit data,
then `python run.py figure7-diversity` to redraw Figure 7. See the new
[release README](appendix_c_geometries/synthetic_diversity_20261105/README.md)
for dependencies, projection domains, normalization and numerical checks.
The self-intersecting figure eight has no added deficit curve. Historical
transport measurements, topology records and plotting code are preserved.

## Historical paper and code map


| Paper location | Experiment | Directory |
|---|---|---|
| Section 3, Figure 1 | Topological phases and circle trajectories | `section_03_topological_phases/` |
| Section 4, Figure 2 | Topology and distributional error | `section_04_distribution/figure02/` |
| Section 5, Figure 3 | Early stopping and denoising | `section_05_early_stopping/` |
| Section 6.1, Figure 4 | Empirical averaging and ESS | `section_06_discussion/figure04/` |
| Section 6.2, Figures 5–6 | Empirical, U-Net, and MLP comparisons | `section_06_discussion/figure05/`, `figure06/` |
| Section 6.3, Figure 7 | Bidirectional finite-window field replacement | `section_06_discussion/figure07/` |
| Appendix B | Time coordinates and numerical protocols | `appendix_b_protocols/` |
| Appendix C, Figure 8 | Four geometries and refinement checks | `appendix_c_geometries/` |
| Appendix D, Figure 9 | MNIST experiments | `appendix_d_images/figure09/` |
| Appendix D, Figure 10 | Extended MLP training comparison | `appendix_d_images/figure10/`, `extended_mlp/` |

Shared numerical routines, models, and preprocessing are under `shared/`.
The experiment settings are in [`configs/paper_protocol.json`](configs/paper_protocol.json).
The main CelebA experiments use **200,000 training steps**; MNIST uses **100,000**.
Figure 10 retains the separate extended-training experiment in the appendix.
Historical directory names can differ from the current figure numbering above.

## Quick start

Use Python 3.12 for the pinned numerical environment:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-numerics.txt
python run.py list
python run.py check-source
python run.py verify-manifest
python run.py smoke
python run.py time
```

`check-source` parses Python sources. `verify-manifest` checks distributed file
hashes. `smoke` tests the numerical geometry and transport routines; `time`
independently integrates the saved trajectories in linear flow-matching time.
These commands do not train a network.

## Figures and reproduction

For the current JMLR typography exports, see
[`publication/README.md`](publication/README.md). The original experiment
renderers below remain available for the historical layouts.

```bash
python run.py figure2
python run.py figure3
python run.py topology
```

Figures 2 and 3 use included synthetic measurements and point clouds. Figure 1
recomputes trajectories; Figure 8 recomputes geometry and assignment calculations.
Image figures (4–7, 9–10) require the corresponding image inputs described in
[Data and models](docs/DATA_AND_MODELS.md). Those image archives are retained in
the private research archive and omitted from the public release. The exporter
lists each omission in `configs/public_export.json`.

Saved numerical measurements remain identified as saved measurements; redrawing
a figure does not rerun training. For compute paths, external inputs, and checks,
see [Reproduction guide](docs/REPRODUCING.md).

GPU experiments use Linux, CUDA, and PyTorch. Install a PyTorch build appropriate
for your CUDA environment before `requirements-gpu.txt`. Its version ranges are
dependency constraints, not a list of tested combinations. Checkpoints are external
inputs and are not distributed in Git. Figure fonts may fall back to local fonts.

## Citation

Please cite the accompanying manuscript when using these experiments:

```bibtex
@unpublished{sun2026topology,
  author = {Sun, Jinghuan and Wang, Jianxing and Zhang, Yaoyu},
  title = {Topology and Distributional Generalization in Empirical Diffusion Models},
  year = {2026},
  note = {Manuscript}
}
```

Machine-readable metadata are in [CITATION.cff](CITATION.cff) and
[citation.bib](citation.bib). The manuscript is currently cited as an unpublished work;
its paper identifier will be added when available.

## License and data

The code and accompanying documentation are licensed under [MIT](LICENSE).
Copyright (c) 2026 Jinghuan Sun, Jianxing Wang, and Yaoyu Zhang.
Dataset and dependency licenses remain separate. See
[third-party notices](THIRD_PARTY_NOTICES.md) and
[data instructions](docs/DATA_AND_MODELS.md).
