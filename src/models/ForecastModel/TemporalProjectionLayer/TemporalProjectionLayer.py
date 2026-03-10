import torch
import torch.nn as nn

class TemporalProjectionLayer(nn.Module):
    def __init__(self, seq_length: int, pred_len: int = 3):
        super().__init__()
        self.linear = nn.Linear(seq_length, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        return x