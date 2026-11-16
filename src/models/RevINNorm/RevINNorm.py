"""Reversible Instance Normalization (RevIN) cho chuỗi đơn biến và đa biến."""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Chuẩn hoá đơn biến (B, L) -> (B, L) bằng mean/std của từng instance."""

    def __init__(self, num_features: int = 1, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=-1, keepdim=True).detach()
        self._std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        out = (x - self._mean) / self._std
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        return x * self._std.view(shape) + self._mean.view(shape)

    def forward(self, x: torch.Tensor, mode: str = "norm"):
        if mode == "norm":
            return self.normalize(x)
        elif mode == "denorm":
            return self.denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")


class RevINMulti(nn.Module):
    """RevIN đa kênh: chuẩn hóa mỗi kênh theo mean/std của chính nó (B, L, F).
    Đảo ngược CHỈ cho kênh mục tiêu (target_idx) khi xuất quantile."""

    def __init__(self, n_features: int, target_idx: int = 0, affine: bool = False,
                 eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.target_idx = target_idx
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta = nn.Parameter(torch.zeros(n_features))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, F)
        self._mean = x.mean(dim=1, keepdim=True).detach()                       # (B, 1, F)
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        out = (x - self._mean) / self._std
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:  # (B, H, Q)
        i = self.target_idx
        if self.affine:
            y = (y - self.beta[i]) / (self.gamma[i] + self.eps)
        m = self._mean[:, :, i:i + 1]      # (B, 1, 1)
        s = self._std[:, :, i:i + 1]
        return y * s + m

    def forward(self, x: torch.Tensor, mode: str = "norm"):
        if mode == "norm":
            return self.normalize(x)
        elif mode == "denorm":
            return self.denormalize_target(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")


# Alias backward compatibility
RevINNorm = RevIN
