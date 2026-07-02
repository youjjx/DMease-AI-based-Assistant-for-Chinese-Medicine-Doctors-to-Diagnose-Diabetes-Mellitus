from __future__ import annotations

import torch
from torch import nn


class SplineEdgeLayer(nn.Module):
    """A compact KAN-style layer with learnable edge-wise univariate functions.

    This is intentionally lightweight. It preserves the paper's KAN interface
    while keeping the repository runnable before the full PPO training data is
    available. A production run can replace this module with pykan or another
    spline implementation without changing policy inputs and outputs.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.knots = nn.Parameter(torch.linspace(-1.0, 1.0, grid_size), requires_grad=False)
        self.coefficients = nn.Parameter(torch.randn(in_features, out_features, grid_size) * 0.02)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(inputs, -1.0, 1.0)
        distances = torch.abs(clipped.unsqueeze(-1) - self.knots)
        basis = torch.relu(1.0 - distances * (self.grid_size - 1) / 2.0)
        spline = torch.einsum("big,iog->bo", basis, self.coefficients)
        return self.linear(inputs) + spline


class KANPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, num_herbs: int, hidden_width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            SplineEdgeLayer(input_dim, hidden_width),
            nn.SiLU(),
            SplineEdgeLayer(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, num_herbs),
        )

    def forward(self, state: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(state)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        return logits

