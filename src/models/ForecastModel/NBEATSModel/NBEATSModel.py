import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.RevINNorm.RevINNorm import RevIN
from src.models.ForecastModel.PredictionHead.QuantileHead import DEFAULT_QUANTILES


class NBEATSBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, n_quantiles: int,
                 hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = lookback
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_size, lookback)
        self.forecast_head = nn.Linear(hidden_size, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        return self.backcast_head(h), self.forecast_head(h)


class NBEATSModel(nn.Module):
    def __init__(self, seq_length: int, pred_len: int = 1,
                 n_stacks: int = 2, n_blocks: int = 2, n_layers: int = 3,
                 layer_dim: int = 256, dropout: float = 0.1,
                 quantiles: list = DEFAULT_QUANTILES, revin_affine: bool = True, **kwargs):
        super().__init__()
        self.seq_length = seq_length
        self.pred_len = pred_len
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        self.revin = RevIN(affine=revin_affine)
        self.blocks = nn.ModuleList([
            NBEATSBlock(seq_length, pred_len, self.n_quantiles, layer_dim, n_layers, dropout)
            for _ in range(n_stacks * n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, :, 0]
        x = self.revin.normalize(x)
        residual = x
        forecast = torch.zeros(x.shape[0], self.pred_len * self.n_quantiles,
                               device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        forecast = forecast.view(x.shape[0], self.pred_len, self.n_quantiles)
        base = forecast[..., :1]
        increments = F.softplus(forecast[..., 1:])
        out = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        return self.revin.denormalize(out)