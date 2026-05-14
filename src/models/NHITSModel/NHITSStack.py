import torch
import torch.nn as nn
from .NHITSBlock import NHITSBlock

class NHITSStack(nn.Module):
    def __init__(self, seq_length: int, pred_len: int, n_features: int,
                 n_blocks: int, n_layers: int, n_theta: int, hidden_dim: int,
                 k_size: int, dropout: float = 0.0):
        super().__init__()
        self.pred_len = pred_len
        self.blocks = nn.ModuleList([
            NHITSBlock(seq_length, pred_len, n_features, n_layers, n_theta, hidden_dim, k_size, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        stack_forecast = torch.zeros(x.size(0), self.pred_len, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            forecast, backcast = block(residual)
            residual = residual - backcast
            stack_forecast = stack_forecast + forecast
        return stack_forecast, residual