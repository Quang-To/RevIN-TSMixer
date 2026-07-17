import torch
import torch.nn as nn

from src.models.ForecastModel.RevINNorm.RevINNorm import RevINNorm
from src.models.ForecastModel.MixingLayer.MixerBlock import MixerBlock
from src.models.ForecastModel.TemporalProjectionLayer.TemporalProjectionLayer import TemporalProjectionLayer

class ForecastModel(nn.Module):
    def __init__(self, seq_length: int, ff_dim: int, dropout: float, pred_len: int, n_block: int, n_features: int = 6):
        super().__init__()
        self.rev_norm = RevINNorm(num_features=n_features)

        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(seq_length=seq_length, ff_dim=ff_dim, dropout=dropout, num_features=n_features) for _ in range(n_block)]
        )
        self.temporal_projection = TemporalProjectionLayer(seq_length=seq_length, pred_len=pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rev_norm(x, mode="norm")
        for block in self.mixer_blocks:
            x = block(x)
        x = self.temporal_projection(x)
        x = self.rev_norm(x, mode="denorm")
        return x[:, :, -1]