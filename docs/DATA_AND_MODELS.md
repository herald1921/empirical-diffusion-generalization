# Data and models

## Included material

The public release includes code, experimental configurations, synthetic data,
CSV/JSON numerical measurements, and the CelebA preprocessing order/index contract.
Image-related NPY/NPZ archives are conservatively excluded as whole files, including
mixed archives containing both images and metrics. This also excludes MNIST image
archives; it does not assert that MNIST has the same redistribution terms as CelebA.
No pretrained weights are included.

The exact file list is generated in `configs/public_export.json`. Original arrays,
historical measurements, source provenance, and Git history remain in the private
research archive. The public release is a separate directory with no inherited Git
history. CSV/JSON values are preserved; private location strings are generalized.

## CelebA

Obtain the dataset from the [official CelebA project](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Its agreement limits use to non-commercial research and restricts redistribution.
This repository's code license does not grant rights to CelebA images.
Image galleries, nearest-training-image arrays, intermediate image states, and
image-bearing frozen plotting inputs are not part of the public release.

The paper preprocessing uses a (9, 29, 169, 189) crop, 80 x 80 resolution, and RGB
averaging for grayscale. The retained count is 196,047, with two groups of 98,023
and one unused middle image. Use `shared/celeba_preprocessing/prepare_celeba80_paper.py`
and `shared/celeba_preprocessing/paper_contract/`; consult `--help` for paths and
options. The contract records the order and removals; it contains no image pixels.
The preparation script checks these inputs rather than silently choosing a new order.

## MNIST

The experiment uses the 60,000-image training set, values divided by 255, split seed
20260924, two groups of 30,000, and prefix subsets of size 100, 1,000, and 10,000.
Supply the original IDX files under `mnist/MNIST/raw/`. Use the provider's dataset
terms independently of the code license.

## Weights and regeneration

Training checkpoints and large raw datasets are external inputs. The paper's
configuration records identify training budgets, architectures, seeds, and readout
conventions. Generate image intermediates locally from authorized data and matching
weights before invoking image figure scripts. This release does not offer a public
checkpoint download. Sharing weights requires a separate decision.
