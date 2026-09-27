import torch
import torch.nn as nn


class TemporalProjectionLayer(nn.Module):
    def __init__(self, seq_length: int, pred_len: int, n_quantiles: int):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = n_quantiles
        self.linear = nn.Linear(seq_length, pred_len * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.linear(x)                                  
        x = x.view(B, self.pred_len, self.n_quantiles)      
        return x