import torch
import torch.nn as nn

from src.models.ForecastModel.MixingLayer.TimeMixingLayer import TimeMixingLayer
from src.models.ForecastModel.MixingLayer.FeatureMixingLayer import FeatureMixingLayer

class MixerBlock(nn.Module):
    def __init__(self, seq_length, ff_dim, dropout):
        super().__init__()
        self.time_mixing = TimeMixingLayer(seq_length, dropout)
        self.feature_mixing = FeatureMixingLayer(ff_dim=ff_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x