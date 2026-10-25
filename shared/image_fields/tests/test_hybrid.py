import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celeba_nn_stage.hybrid import integrate_experiment1_pair


class OnesModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.ones_like(x)


def test_experiment1_switch_constant_fields():
    x0 = torch.zeros(2, 4)
    model = OnesModel()

    def empirical_zero(x, t):
        return torch.zeros_like(x)

    full, hybrid, cap_full, cap_hybrid = integrate_experiment1_pair(
        model,
        x0,
        empirical_velocity=empirical_zero,
        switch_t=0.8,
        steps=100,
        capture_t=(0.8, 1.0),
        autocast_bf16=False,
    )
    assert torch.allclose(full, torch.ones_like(full), atol=1e-5)
    assert torch.allclose(hybrid, torch.full_like(hybrid, 0.8), atol=1e-5)
    assert torch.allclose(cap_full[0.8], cap_hybrid[0.8], atol=1e-6)
    assert torch.allclose(cap_hybrid[1.0], hybrid.cpu(), atol=1e-6)

from celeba_nn_stage.hybrid import integrate_field_midpoint, integrate_nn_suffix


def test_experiment2_empirical_prefix_then_nn_suffix_constant_fields():
    x0 = torch.zeros(2, 4)

    def empirical_one(x, t):
        return torch.ones_like(x)

    prefix_final, prefix_caps = integrate_field_midpoint(
        x0,
        field=empirical_one,
        start_t=0.0,
        end_t=0.8,
        steps=100,
        capture_t=(0.8,),
    )
    assert torch.allclose(prefix_final, torch.full_like(prefix_final, 0.8), atol=1e-5)
    assert torch.allclose(prefix_caps[0.8], prefix_final.cpu(), atol=1e-6)

    model = OnesModel()
    suffix_final, suffix_caps = integrate_nn_suffix(
        model,
        prefix_final,
        switch_t=0.8,
        steps=100,
        capture_t=(0.8, 1.0),
        autocast_bf16=False,
    )
    assert torch.allclose(suffix_final, torch.ones_like(suffix_final), atol=1e-5)
    assert torch.allclose(suffix_caps[0.8], prefix_final.cpu(), atol=1e-6)
    assert torch.allclose(suffix_caps[1.0], suffix_final.cpu(), atol=1e-6)
