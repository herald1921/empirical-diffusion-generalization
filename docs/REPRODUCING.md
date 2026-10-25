# Reproducing the experiments

## Numerical checks

Run commands from the repository root. The numerical dependencies are pinned in
`requirements-numerics.txt`. The complete GPU environment additionally needs the
dependencies in `requirements-gpu.txt`.

| Command | Work performed | Inputs |
|---|---|---|
| `python run.py smoke` | Geometry and optimal-transport sanity checks | Included code |
| `python run.py time` | Independent trajectory integration | Included synthetic NPZ |
| `python run.py topology` | Exact alpha-complex topology diagnostics | Included synthetic records |
| `python run.py geometry` | Geometry/refinement diagnostics | Included synthetic records; potentially expensive |
| `python -m unittest discover -s tests -p test_repairs.py` | Exact nearest-neighbor and data-contract regression tests | PyTorch, no full dataset |
| `python run.py mnist-metrics` | MNIST saved-input checks | Private image archives; some external-source checks may skip |

The repair tests use synthetic queries, direct float64 distance calculations, and
controlled data fixtures. They do not validate trained model quality.

## Figure generation

`python run.py figureN` selects one figure. Figures 2–7 and 9–10 primarily reuse
saved measurements; Figure 2 also checks paired integration. Figure 1 integrates
trajectories and Figure 8 computes geometric/assignment results. Figure 6 uses the
reconstructed plotting entry point distributed with the repository.

The aggregate `python run.py figures` runs Figures 2–10, so it requires the private
image inputs. It is not a public-release quick-start command. Generated products
usually go to each figure's `output/` or `revised/` folder and are ignored by Git.

## Neural experiments

1. Obtain CelebA and MNIST from their respective providers under their own terms.
2. Prepare datasets following `docs/DATA_AND_MODELS.md` and the paper protocol.
3. Use a **new** computation directory to relocate the historical Linux sources:

```bash
python tools/prepare_compute.py --work-dir /absolute/path/to/new-work --datasets /absolute/path/to/datasets
```

The dataset directory contains `celeba80_author_order/` and/or `mnist/MNIST/raw/`.
This command copies sources and rewrites workspace paths; it starts no computation.
The export uses generic historical workspace names, which the same tool relocates.
Inspect `prepared_sources.json` for the prepared locations. Provide matching
checkpoints in the run/checkpoint locations needed by the selected evaluator.
Preserve source-integrity checks when resuming an existing run.

| Task | Source entry point |
|---|---|
| CelebA training | `section_06_discussion/celeba/training/code/run.py` |
| CelebA pixel-distance evaluation | `section_06_discussion/celeba/metrics/code/` |
| Full Figure 5 input collection | `section_06_discussion/figure05/collect_inputs.py` |
| Shared-input field probes | `section_06_discussion/common_inputs/code/probes.py` |
| Field replacement | `section_06_discussion/field_replacement/evaluation/` |
| MNIST training | `appendix_d_images/mnist/code/run_mnist.py` |
| MNIST evaluation | `appendix_d_images/mnist/evaluation/evaluate_missing.py` |
| Extended MLP experiment | `appendix_d_images/extended_mlp/training/` |

These are research entry points with run-specific inputs, not a single end-to-end
training command. Follow their argument definitions and the saved configuration.
Archive new runs separately from the paper records.

## Interpretation

CelebA's reference groups are image-disjoint; no identity-disjoint claim is made.
The main protocol uses one initialization per configuration, paired inputs for
comparisons, and the metric/readout conventions in `configs/paper_protocol.json`.
Preserve these choices when comparing reproduced numbers. The field-replacement
output is the integrated state at T=0.99 without terminal denoising.

Hash manifests establish integrity of the distributed files. They do not establish
that a new dataset download, training environment, or stochastic run is identical.
