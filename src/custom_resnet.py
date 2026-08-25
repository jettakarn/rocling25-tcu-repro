from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.norm(self.fc(x))))
        return x + h


class CustomResNet(nn.Module):
    """Paper-style residual MLP regressor for joint VA prediction.

    Architecture follows Li & Lin (ROCLING 2025): RMSNorm → 6 residual
    blocks (FC + RMSNorm + ReLU + Dropout) → two FC projection layers →
    ``sin(x) * 4 + 5`` to map into the 1–9 rating range.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 6,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.in_norm = RMSNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_norm(self.in_proj(x))
        for block in self.blocks:
            h = block(h)
        h = torch.relu(self.fc1(h))
        logits = self.fc2(h)
        return torch.sin(logits) * 4.0 + 5.0
