import torch
import torch.nn as nn
import torch.nn.functional as F


class NHITSBlock(nn.Module):
    def __init__(self, seq_length: int, pred_len: int, n_features: int, n_layers: int,
        n_theta: int, hidden_dim: int, k_size: int, dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.seq_length = seq_length
        self.pooling = nn.MaxPool1d(kernel_size=k_size, stride=k_size)
        with torch.no_grad():
            dummy = torch.zeros(1, n_features, seq_length)
            pooled_len = self.pooling(dummy).shape[-1]
        in_dim = pooled_len * n_features
        layers = []
        cur = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(cur, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.fc_backcast = nn.Linear(hidden_dim, seq_length * n_features)
        self.fc_forecast  = nn.Linear(hidden_dim, n_theta)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) != self.n_features and x.size(2) == self.n_features:
            x = x.transpose(1, 2)

        B = x.size(0)
        h = self.pooling(x)
        h = h.reshape(B, -1)
        h = self.mlp(h)
        backcast = self.fc_backcast(h).reshape(B, self.n_features, self.seq_length)
        knots = self.fc_forecast(h)[:, None, :]
        forecast = F.interpolate(knots, size=self.pred_len, mode='linear', align_corners=False)
        forecast = forecast[:, 0, :]
        return forecast, backcast