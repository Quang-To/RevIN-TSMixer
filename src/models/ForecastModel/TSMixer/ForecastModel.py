import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.RevINNorm.RevINNorm import RevINMulti
from src.models.ForecastModel.PredictionHead.QuantileHead import DEFAULT_QUANTILES, monotone_quantiles


class _TSMixerBlock(nn.Module):
    def __init__(self, lookback: int, n_channels: int, hidden: int, dropout: float, norm: str = "layer"):
        super().__init__()
        self.norm = norm
        if norm == "batch":
            self.norm1 = nn.BatchNorm2d(1)
            self.norm2 = nn.BatchNorm2d(1)
        else:
            self.norm1 = nn.LayerNorm(n_channels)
            self.norm2 = nn.LayerNorm(n_channels)
        self.time = nn.Linear(lookback, lookback)
        self.feat1 = nn.Linear(n_channels, hidden)
        self.feat2 = nn.Linear(hidden, n_channels)
        self.drop = nn.Dropout(dropout)

    def _apply_norm(self, layer, x):  # (B, L, C)
        if self.norm == "batch":
            return layer(x.unsqueeze(1)).squeeze(1)
        return layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time-mixing
        h = self._apply_norm(self.norm1, x).transpose(1, 2)   # (B, C, L)
        h = self.drop(F.relu(self.time(h))).transpose(1, 2)
        x = x + h
        # Feature-mixing
        h = self._apply_norm(self.norm2, x)
        h = self.drop(self.feat2(F.relu(self.feat1(h))))
        return x + h


class _TSMixerCoreMulti(nn.Module):
    def __init__(self, lookback: int, n_features: int, horizon: int, n_quantiles: int,
                 hidden: int, n_layers: int, dropout: float, norm: str = "layer"):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.blocks = nn.ModuleList([
            _TSMixerBlock(lookback, n_features, hidden, dropout, norm=norm)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(lookback * n_features, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, F)
        for blk in self.blocks:
            x = blk(x)
        h = x.reshape(x.shape[0], -1)
        return monotone_quantiles(self.head(h), self.horizon, self.n_quantiles)


class ForecastModel(nn.Module):
    """TSMixer đa biến có RevINMulti: nhận (B, L, F) xuất (B, H, Q)."""

    def __init__(self, seq_length: int, ff_dim: int, dropout: float, pred_len: int = 1,
                 n_block: int = 3, n_features: int = 7, quantiles: list = DEFAULT_QUANTILES,
                 target_idx: int = 0, revin_affine: bool = False, norm: str = "layer", **kwargs):
        super().__init__()
        self.seq_length = seq_length
        self.pred_len = pred_len
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.target_idx = target_idx

        self.revin = RevINMulti(n_features, target_idx=target_idx, affine=revin_affine)
        self.core = _TSMixerCoreMulti(
            lookback=seq_length,
            n_features=n_features,
            horizon=pred_len,
            n_quantiles=self.n_quantiles,
            hidden=ff_dim,
            n_layers=n_block,
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.revin.normalize(x)
        out = self.core(xn)
        return self.revin.denormalize_target(out)