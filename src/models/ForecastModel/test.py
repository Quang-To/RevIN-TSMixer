import torch
from src.models.ForecastModel.ForecastModel import ForecastModel

B = 32
T = 96
C = 5
N = 24

model = ForecastModel(
    seq_length=T,
    num_features=C,
    ff_dim=64,
    dropout=0.1,
    pred_len=N,
    n_block=4
)

x = torch.randn(B, T, C)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)