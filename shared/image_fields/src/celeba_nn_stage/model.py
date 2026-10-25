from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 48
    time_emb_dim: int = 256
    dropout: float = 0.0
    time_scale: float = 1000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for official FM time t in [0,1]."""

    def __init__(self, dim: int, time_scale: float = 1000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time_emb_dim must be even")
        half = dim // 2
        exponent = -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        self.register_buffer("freqs", torch.exp(exponent), persistent=False)
        self.time_scale = float(time_scale)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = (t.float() * self.time_scale)[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return self.proj(emb)


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise RuntimeError(f"No valid GroupNorm group count for {channels}")


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class VelocityUNet80(nn.Module):
    """80x80 grayscale U-Net for conditional flow matching.

    Official time axis: t in [0,1].
    Path: x_t = (1-t) x_0 + t x_1, x_0 ~ N(0,I), x_1 ~ empirical S1.
    Target velocity: x_1 - x_0.
    Sigma is NOT an input/training variable; sigma=(1-t)/t is only a secondary
    coordinate used later to compare with the previous VE-style experiments.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        b, e, d = config.base_channels, config.time_emb_dim, config.dropout
        self.time_embed = SinusoidalTimeEmbedding(e, config.time_scale)
        self.in_conv = nn.Conv2d(1, b, 3, padding=1)

        self.enc1a = ResBlock(b, b, e, d)
        self.enc1b = ResBlock(b, b, e, d)
        self.down1 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)

        self.enc2a = ResBlock(2 * b, 2 * b, e, d)
        self.enc2b = ResBlock(2 * b, 2 * b, e, d)
        self.down2 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)

        self.enc3a = ResBlock(4 * b, 4 * b, e, d)
        self.enc3b = ResBlock(4 * b, 4 * b, e, d)
        self.down3 = nn.Conv2d(4 * b, 4 * b, 4, stride=2, padding=1)

        self.mid1 = ResBlock(4 * b, 4 * b, e, d)
        self.mid2 = ResBlock(4 * b, 4 * b, e, d)

        self.up3 = nn.ConvTranspose2d(4 * b, 4 * b, 4, stride=2, padding=1)
        self.dec3a = ResBlock(8 * b, 4 * b, e, d)
        self.dec3b = ResBlock(4 * b, 4 * b, e, d)

        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 4, stride=2, padding=1)
        self.dec2a = ResBlock(4 * b, 2 * b, e, d)
        self.dec2b = ResBlock(2 * b, 2 * b, e, d)

        self.up1 = nn.ConvTranspose2d(2 * b, b, 4, stride=2, padding=1)
        self.dec1a = ResBlock(2 * b, b, e, d)
        self.dec1b = ResBlock(b, b, e, d)

        self.out_norm = _group_norm(b)
        self.out_conv = nn.Conv2d(b, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_flat: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        if x_flat.ndim == 2:
            x = x_flat.reshape(-1, 1, 80, 80)
        elif x_flat.ndim == 4 and x_flat.shape[1:] == (1, 80, 80):
            x = x_flat
        else:
            raise ValueError(f"Expected [B,6400] or [B,1,80,80], got {tuple(x_flat.shape)}")

        t_tensor = torch.as_tensor(t, device=x.device, dtype=torch.float32)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(x.shape[0])
        if t_tensor.shape != (x.shape[0],):
            raise ValueError("t must be scalar or shape [batch]")
        if bool(((t_tensor < 0) | (t_tensor > 1)).any()):
            raise ValueError("t must lie in [0,1]")

        emb = self.time_embed(t_tensor)
        h = self.in_conv(x)
        s1 = self.enc1b(self.enc1a(h, emb), emb)
        h = self.down1(s1)
        s2 = self.enc2b(self.enc2a(h, emb), emb)
        h = self.down2(s2)
        s3 = self.enc3b(self.enc3a(h, emb), emb)
        h = self.down3(s3)
        h = self.mid2(self.mid1(h, emb), emb)
        h = self.up3(h)
        h = self.dec3b(self.dec3a(torch.cat([h, s3], dim=1), emb), emb)
        h = self.up2(h)
        h = self.dec2b(self.dec2a(torch.cat([h, s2], dim=1), emb), emb)
        h = self.up1(h)
        h = self.dec1b(self.dec1a(torch.cat([h, s1], dim=1), emb), emb)
        out = self.out_conv(F.silu(self.out_norm(h)))
        return out.reshape(out.shape[0], -1)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device, use_ema: bool = True) -> VelocityUNet80:
    cfg = ModelConfig(**checkpoint["model_config"])
    model = VelocityUNet80(cfg).to(device)
    key = "ema_model" if use_ema and "ema_model" in checkpoint else "model"
    model.load_state_dict(checkpoint[key])
    model.eval()
    return model
