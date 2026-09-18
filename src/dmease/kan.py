from __future__ import annotations

import torch
from torch import nn


class SplineEdgeLayer(nn.Module):
    """A compact KAN-style layer with learnable edge-wise univariate functions.

    The module preserves the paper's KAN interface and can be replaced with
    another spline implementation without changing policy inputs and outputs.
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


class KANLayer(nn.Module):
    """Edge-function KAN layer using trainable Gaussian basis functions."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, grid_size))
        self.coefficients = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, mean=0.0, std=0.04)
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-8.0 * (inputs.unsqueeze(-1) - self.grid) ** 2)
        nonlinear = torch.einsum("big,oig->bo", basis, self.coefficients)
        base = torch.nn.functional.silu(inputs) @ self.base_weight.t()
        return nonlinear + base + self.bias


class KANPolicy(nn.Module):
    """KAN actor-critic used by the PPO training workflow."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32, grid_size: int = 8):
        super().__init__()
        self.actor = nn.Sequential(
            KANLayer(state_dim, hidden_dim, grid_size),
            nn.LayerNorm(hidden_dim),
            KANLayer(hidden_dim, action_dim, grid_size),
        )
        self.critic = nn.Sequential(
            KANLayer(state_dim, hidden_dim, grid_size),
            nn.LayerNorm(hidden_dim),
            KANLayer(hidden_dim, 1, grid_size),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(state), self.critic(state).squeeze(-1)

    def distribution(self, state: torch.Tensor, action_mask: torch.Tensor | None = None):
        logits, value = self(state)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits), value

