"""
ClimateUNet - the high-end spatiotemporal forecaster (PyTorch / CUDA).

Design choices that matter for ISRO BAH:
  * Predicts ANOMALIES vs day-of-year climatology (stable multi-day rollout,
    meaningful skill) rather than raw fields.
  * DIRECT multi-horizon: one forward pass emits all HORIZON lead days, so there
    is no autoregressive feedback loop to diverge (the failure mode of the old
    ConvLSTM).
  * Residual U-Net with squeeze-excite channel attention - modern, accurate, and
    runs comfortably on a 6 GB RTX 4050.
  * Input = INPUT_DAYS x 3 variables stacked as channels (time-as-channels, the
    SimVP insight) -> output = HORIZON x 3 variables.

Tensors are channels-first (N, C, H, W) per PyTorch convention.
Input  : (N, INPUT_DAYS*3, H, W)   scaled anomalies
Output : (N, HORIZON*3,   H, W)    scaled anomalies
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, ch, ratio=8):
        super().__init__()
        hidden = max(ch // ratio, 4)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):
        s = x.mean(dim=(2, 3))               # global avg pool -> (N, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """Residual conv block: GroupNorm + GELU + SE attention."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        g = lambda c: min(8, c)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g(out_ch), out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        return F.gelu(self.proj(x) + h)


class ClimateUNet(nn.Module):
    """Residual forecaster: refines a persistence-of-anomaly (POA) prior.

    Input  : (N, INPUT_DAYS*3 + HORIZON*3, H, W)
             = [history anomalies] ++ [POA prior anomalies]
    Output : (N, HORIZON*3, H, W) = POA_prior + learned_residual

    Because the POA prior is added back as a skip connection, the network only
    has to learn the *correction*, so it matches POA in the worst case and beats
    it when it finds structure - exactly what separates it from a naive model.
    """

    def __init__(self, input_days=C.INPUT_DAYS, horizon=C.HORIZON, n_vars=3,
                 n_drivers=0, base=48, pad_to=144,
                 grid=(C.GRID_NLAT, C.GRID_NLON), dropout=0.2):
        super().__init__()
        self.H, self.W = grid
        self.pad_to = pad_to
        self.hist_ch = input_days * n_vars          # 21
        self.drv_ch = input_days * n_drivers        # 0 (v1) or 28 (v2, K=4)
        self.poa_ch = horizon * n_vars              # 30
        self.in_ch = self.hist_ch + self.drv_ch + self.poa_ch   # 51 / 79
        self.out_ch = horizon * n_vars              # 30
        self.n_drivers = n_drivers

        self.stem = nn.Conv2d(self.in_ch, base, 3, padding=1)
        self.e1 = ResBlock(base, base)
        self.e2 = ResBlock(base, base * 2)
        self.e3 = ResBlock(base * 2, base * 4)
        self.pool = nn.AvgPool2d(2)
        self.b1 = ResBlock(base * 4, base * 8)
        self.b2 = ResBlock(base * 8, base * 8)
        self.drop = nn.Dropout2d(dropout)
        self.d3 = ResBlock(base * 8 + base * 4, base * 4)
        self.d2 = ResBlock(base * 4 + base * 2, base * 2)
        self.d1 = ResBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, self.out_ch, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # start at POA

    def forward(self, x):
        poa = x[:, self.hist_ch + self.drv_ch:, :self.H, :self.W]  # POA prior
        ph, pw = self.pad_to - self.H, self.pad_to - self.W
        z = F.pad(x, (0, pw, 0, ph))
        z = F.gelu(self.stem(z))

        e1 = self.e1(z)                              # 144
        e2 = self.e2(self.pool(e1))                  # 72
        e3 = self.e3(self.pool(e2))                  # 36
        b = self.b2(self.b1(self.pool(e3)))          # 18
        b = self.drop(b)

        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(b), e3], dim=1))  # 36
        d2 = self.d2(torch.cat([up(d3), e2], dim=1)) # 72
        d1 = self.d1(torch.cat([up(d2), e1], dim=1)) # 144
        residual = self.head(d1)[:, :, :self.H, :self.W]
        return residual + poa                        # refine the POA prior


def build_model(**kw):
    return ClimateUNet(**kw)


if __name__ == "__main__":
    m = build_model()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, C.INPUT_DAYS * 3 + C.HORIZON * 3, C.GRID_NLAT, C.GRID_NLON)
    y = m(x)
    print(f"params: {n:,}")
    print("in", tuple(x.shape), "out", tuple(y.shape))
    print("cuda available:", torch.cuda.is_available())
