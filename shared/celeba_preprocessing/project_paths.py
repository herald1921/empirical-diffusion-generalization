"""Portable paths for the CelebA80 experiments."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", PROJECT_DIR.parents[1] if len(PROJECT_DIR.parents) > 1 else PROJECT_DIR))
CELEBA80_DATA_DIR = Path(os.environ.get("CELEBA80_DATA_DIR", WORKSPACE_ROOT / "datasets" / "celeba80_author_order"))
CELEBA80_RUNS_DIR = Path(os.environ.get("CELEBA80_RUNS_DIR", WORKSPACE_ROOT / "runs" / "celeba80_figure2"))
ORACLE_RESULTS_DIR = Path(os.environ.get("CELEBA80_ORACLE_RESULTS_DIR", CELEBA80_RUNS_DIR / "oracle"))
EARLYSTOP_RESULTS_DIR = Path(os.environ.get("CELEBA80_EARLYSTOP_RESULTS_DIR", CELEBA80_RUNS_DIR / "earlystop_sigma_ode"))
DENSITY_PCA_RESULTS_DIR = Path(os.environ.get("CELEBA80_DENSITY_PCA_RESULTS_DIR", CELEBA80_RUNS_DIR / "density_pca_sigma_ode"))
FREQUENCY_PCA_RESULTS_DIR = Path(os.environ.get("CELEBA80_FREQUENCY_PCA_RESULTS_DIR", CELEBA80_RUNS_DIR / "frequency_pca_sigma_ode"))
OUTPUTS_DIR = Path(os.environ.get("CELEBA80_OUTPUTS_DIR", PROJECT_DIR / "outputs"))
FIGURES_DIR = Path(os.environ.get("CELEBA80_FIGURES_DIR", OUTPUTS_DIR / "figures"))
TABLES_DIR = Path(os.environ.get("CELEBA80_TABLES_DIR", OUTPUTS_DIR / "tables"))

def figure_dir(name: str) -> Path:
    path = FIGURES_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
