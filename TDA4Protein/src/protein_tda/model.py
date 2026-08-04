"""Neural network models for interface prediction."""

from __future__ import annotations

import torch
from torch import nn


class InterfaceMLP(nn.Module):
    """Feed-forward neural network used for PCA-reduced TDA/chemical descriptors."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (40, 20, 5, 4), output_dim: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
