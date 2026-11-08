import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.RevINNorm.RevINNorm import RevIN
from src.models.ForecastModel.PredictionHead.QuantileHead import DEFAULT_QUANTILES, monotone_quantiles


def _mlp(in_dim: int, hidden: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    return nn.Sequential(*layers)


class _NHITSBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, n_quantiles: int,
                 hidden: int, n_layers: int, dropout: float, pool_kernel: int):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.pool = nn.MaxPool1d(pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_len = -(-lookback // pool_kernel)
        self.mlp = _mlp(pooled_len, hidden, n_layers, dropout)
        self.backcast_head = nn.Linear(hidden, pooled_len)
        self.forecast_head = nn.Linear(hidden, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # x: (B, L)
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)                         # (B, Lp)
        h = self.mlp(pooled)
        bc_low = self.backcast_head(h).unsqueeze(1)                           # (B, 1, Lp)
        backcast = F.interpolate(bc_low, size=self.lookback, mode="linear",
                                 align_corners=False).squeeze(1)
        forecast = self.forecast_head(h)                                      # (B, H*Q)
        return backcast, forecast


class NHITSModel(nn.Module):
    def __init__(self, seq_length: int, pred_len: int = 1,
                 hidden_dim: int = 256, n_layers: int = 3, dropout: float = 0.1,
                 n_stacks: int = 3, n_blocks: int = 1, pool_kernels: tuple | None = None,
                 quantiles: list = DEFAULT_QUANTILES, revin_affine: bool = True, **kwargs):
        super().__init__()
        self.seq_length = seq_length
        self.pred_len = pred_len
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        # Mỗi stack ứng với một kernel pooling và gồm n_blocks block. Một stack dùng kernel 2,
        # vì kernel 1 (không pooling) làm mô hình trùng hệt N-BEATS.
        if pool_kernels is None:
            pool_kernels = (2,) if n_stacks == 1 else tuple(2 ** i for i in range(n_stacks))
        self.revin = RevIN(affine=revin_affine)
        self.blocks = nn.ModuleList([
            _NHITSBlock(seq_length, pred_len, self.n_quantiles, hidden_dim, n_layers, dropout, k)
            for k in pool_kernels for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, :, 0]
        x = self.revin.normalize(x)
        residual = x
        forecast = torch.zeros(x.shape[0], self.pred_len * self.n_quantiles,
                               device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            backcast, block_forecast = blk(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        out = monotone_quantiles(forecast, self.pred_len, self.n_quantiles)
        return self.revin.denormalize(out)
