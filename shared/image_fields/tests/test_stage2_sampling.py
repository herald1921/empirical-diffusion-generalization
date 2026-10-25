import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from celeba_nn_stage.sampling import make_fixed_noise, integrate_heun

class ConstantField(torch.nn.Module):
    def forward(self, x, t):
        return torch.ones_like(x) * 2.0

def test_fixed_noise_repeatable():
    a = make_fixed_noise(3, 7, 123)
    b = make_fixed_noise(3, 7, 123)
    assert torch.equal(a,b)

def test_heun_constant_field():
    x0 = torch.zeros(2,5)
    x1, caps = integrate_heun(ConstantField(), x0, steps=10, capture_t=[0.0,0.5,1.0], autocast_bf16=False)
    assert torch.allclose(x1, torch.full_like(x1,2.0), atol=1e-6)
    assert torch.allclose(caps[0.5], torch.ones_like(caps[0.5]), atol=1e-6)
