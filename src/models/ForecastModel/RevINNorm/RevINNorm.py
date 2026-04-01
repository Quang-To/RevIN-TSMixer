import torch
import torch.nn as nn

class RevINNorm(nn.Module):
    def __init__(self, num_features: int = 6, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.mean = None
        self.std = None

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True, unbiased=False)
            x = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x = x * self.gamma + self.beta
            return x

        elif mode == "denorm":
            if self.mean is None or self.std is None:
                raise RuntimeError("norm must be called before denorm")
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * (self.std + self.eps) + self.mean
            return x
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")