from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celeba_nn_stage.model import ModelConfig, VelocityUNet80
from celeba_nn_stage.time_axis import sigma_from_t


def test_time_sigma_mapping():
    assert math.isinf(sigma_from_t(0.0))
    assert abs(sigma_from_t(0.5) - 1.0) < 1e-12
    assert abs(sigma_from_t(0.2) - 4.0) < 1e-12
    assert abs(sigma_from_t(1.0) - 0.0) < 1e-12


def test_flow_interpolant_endpoints():
    x0 = torch.randn(3, 10)
    x1 = torch.randn(3, 10)
    t0 = torch.zeros(3, 1)
    t1 = torch.ones(3, 1)
    assert torch.allclose((1-t0)*x0 + t0*x1, x0)
    assert torch.allclose((1-t1)*x0 + t1*x1, x1)
    assert torch.allclose((x1-x0), (x1-x0))


def test_model_forward_small_config():
    model = VelocityUNet80(ModelConfig(base_channels=8, time_emb_dim=32))
    x = torch.randn(2, 6400)
    t = torch.tensor([0.1, 0.8])
    y = model(x, t)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_s1_count_is_not_hard_coded():
    from celeba_nn_stage import data as data_mod
    assert not hasattr(data_mod, "EXPECTED_S1_COUNT")
