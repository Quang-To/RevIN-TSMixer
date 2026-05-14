from typing import List
import torch
import torch.nn as nn
from .NHITSStack import NHITSStack
from src.models.ForecastModel.RevINNorm.RevINNorm import RevINNorm

class NHITSModel(nn.Module):
    def __init__(self, seq_length: int, pred_len: int, n_features: int,
                 n_stacks: int, n_blocks: int, n_layers: int,
                 hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pred_len = pred_len
        self.n_stacks = n_stacks
        self.rev_norm = RevINNorm(num_features=1)
        kernel_list = list(range(n_stacks, 0, -1))
        n_theta_list = [max(pred_len // k, 1) for k in kernel_list]
        self.stacks = nn.ModuleList([
            NHITSStack(seq_length, pred_len, n_features, n_blocks, n_layers, n_theta_list[i], hidden_dim, kernel_list[i], dropout)
            for i in range(n_stacks)
        ])
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        target = x[:, :, -1:].contiguous()
        target_norm = self.rev_norm(target, mode="norm")

        # NHITSBlock expects (batch, channels, time)
        residual = target_norm.transpose(1, 2)
        forecast = torch.zeros(x.size(0), self.pred_len, device=x.device, dtype=x.dtype)
        for stack in self.stacks:
            stack_forecast, residual = stack(residual)
            forecast = forecast + stack_forecast

        forecast = self.rev_norm(forecast.unsqueeze(-1), mode="denorm").squeeze(-1)
        
        return forecast

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)