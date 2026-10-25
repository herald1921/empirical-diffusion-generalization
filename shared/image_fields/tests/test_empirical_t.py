import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celeba_nn_stage.empirical_t import sigma_from_t, empirical_velocity_from_mean, empirical_velocity_t


def test_t_sigma_reference_points():
    assert abs(sigma_from_t(0.5) - 1.0) < 1e-12
    assert abs(sigma_from_t(0.2) - 4.0) < 1e-12
    assert abs(sigma_from_t(0.1) - 9.0) < 1e-12


def test_empirical_velocity_formula():
    x = torch.tensor([[1.0, 2.0]])
    mean = torch.tensor([[3.0, 6.0]])
    v = empirical_velocity_from_mean(x, mean, 0.5)
    assert torch.allclose(v, torch.tensor([[4.0, 8.0]]))


def test_empirical_velocity_t_zero_is_uniform_posterior():
    class DummyCore:
        pass

    ref = torch.tensor([[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]])
    x0 = torch.tensor([[0.5, -0.5], [1.5, 0.5]])
    ref_norms = (ref * ref).sum(1)
    v, post = empirical_velocity_t(
        core=DummyCore(),
        x_t=x0,
        reference=ref,
        reference_norms=ref_norms,
        reference_mean=ref.mean(0),
        t=0.0,
    )
    expected = ref.mean(0).unsqueeze(0) - x0
    assert torch.allclose(v, expected)
    assert torch.allclose(post["ess"], torch.full((2,), 3.0))
    assert torch.allclose(post["top_probability"], torch.full((2,), 1.0 / 3.0))
