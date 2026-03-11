import torch
import torch.nn as nn

class FeatureMixingLayer(nn.Module):
    def __init__(self, ff_dim: int, dropout: float, num_features: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_features),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = x + residual
        return x