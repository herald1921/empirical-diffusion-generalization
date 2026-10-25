# Third-party material

The project code license does not replace third-party terms.

## Dependencies

The dependencies are installed separately, not vendored in this repository.
Retain their notices when redistributing a packaged environment. Consult the
licenses shipped with the exact installed versions; the following are upstream
source locations, not a claim to relicense those packages:

| Dependency | Upstream |
|---|---|
| NumPy | https://github.com/numpy/numpy |
| SciPy | https://github.com/scipy/scipy |
| Matplotlib | https://github.com/matplotlib/matplotlib |
| GUDHI | https://github.com/GUDHI/gudhi-devel |
| Python Optimal Transport (POT) | https://github.com/PythonOT/POT |
| PyTorch / torchvision | https://github.com/pytorch/pytorch / https://github.com/pytorch/vision |
| Pillow | https://github.com/python-pillow/Pillow |
| LPIPS | https://github.com/richzhang/PerceptualSimilarity |
| pytest | https://github.com/pytest-dev/pytest |

## Datasets and model assets

CelebA and MNIST are external datasets. Their terms are independent of the project
code license. See [Data and models](docs/DATA_AND_MODELS.md). Downloaded pretrained
assets (including perceptual-network weights) retain their upstream terms.

## Source and method acknowledgments

The experiment implementations use standard residual networks, sinusoidal time
embeddings, flow matching, numerical integration, and optimal-transport methods.
Scientific references are given in the accompanying manuscript. The libraries
above are called through their public APIs and are installed separately; their
source code and downloaded model assets are not bundled in this release.

The release source review identified no vendored third-party implementation
requiring an additional source-license notice. Please report any overlooked source
attribution with the affected file and upstream reference through GitHub Issues.
