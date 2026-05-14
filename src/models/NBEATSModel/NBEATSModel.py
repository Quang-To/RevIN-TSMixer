import torch
import torch.nn as nn
from src.models.ForecastModel.RevINNorm.RevINNorm import RevINNorm

class NBEATSBlock(nn.Module):
    """
    Single N-BEATS block for a univariate target series.
    Uses a residual MLP over the lookback window to produce backcast and forecast basis terms.
    """
    def __init__(self, seq_length: int, pred_len: int, n_features: int, n_layers: int, layer_dim: int, dropout: float = 0.0):
        super().__init__()
        self.seq_length = seq_length
        self.pred_len   = pred_len
        self.n_features = n_features

        input_dim = seq_length
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, layer_dim),
                nn.ReLU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = layer_dim

        self.fc_layers = nn.Sequential(*layers)

        self.basis_forecast = nn.Linear(layer_dim, pred_len)
        self.basis_backcast = nn.Linear(layer_dim, seq_length)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_length, 1)
        Returns:
            forecast: (batch, pred_len)
            backcast: (batch, seq_length, 1)
        """
        h = x.squeeze(-1)                
        h = self.fc_layers(h)            

        forecast = self.basis_forecast(h)  
        backcast = self.basis_backcast(h).unsqueeze(-1)  

        return forecast, backcast


class NBEATSModel(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.
    Forecasts the target series in the last feature column.
    """
    def __init__(
        self,
        seq_length: int,
        pred_len:   int,
        n_features: int = 6,
        n_stacks:   int = 3,
        n_layers:   int = 4,
        layer_dim:  int = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.pred_len   = pred_len
        self.n_features = n_features
        self.rev_norm   = RevINNorm(num_features=n_features)

        self.stacks = nn.ModuleList([
            NBEATSBlock(seq_length, pred_len, n_features, n_layers, layer_dim, dropout)
            for _ in range(n_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, n_features)
        Returns:
            forecast: (batch, pred_len)
        """
        x        = self.rev_norm(x, mode="norm")
        target   = x[:, :, -1:].contiguous()
        residual = target.clone()
        forecast = torch.zeros(x.size(0), self.pred_len, device=x.device, dtype=x.dtype)

        for block in self.stacks:
            block_forecast, backcast = block(residual)
            forecast = forecast + block_forecast
            residual = residual - backcast  

        if self.rev_norm.affine:
            target_gamma = self.rev_norm.gamma[:, :, -1]
            target_beta = self.rev_norm.beta[:, :, -1]
            forecast = (forecast.unsqueeze(-1) - target_beta) / target_gamma
            forecast = forecast.squeeze(-1)

        if self.rev_norm.mean is None or self.rev_norm.std is None:
            raise RuntimeError("RevIN normalization statistics are unavailable")

        target_mean = self.rev_norm.mean[:, :, -1].squeeze(1)
        target_std = self.rev_norm.std[:, :, -1].squeeze(1)
        forecast = forecast * (target_std + self.rev_norm.eps).unsqueeze(-1)
        forecast = forecast + target_mean.unsqueeze(-1)

        return forecast