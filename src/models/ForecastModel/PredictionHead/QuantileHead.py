import torch
import torch.nn.functional as F

DEFAULT_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def monotone_quantiles(raw: torch.Tensor, horizon: int, n_quantiles: int) -> torch.Tensor:
    """Biến (batch, horizon*n_quantiles) -> (batch, horizon, n_quantiles) đơn điệu."""
    f = raw.view(raw.shape[0], horizon, n_quantiles)
    base = f[..., :1]
    increments = F.softplus(f[..., 1:])
    return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)


def non_crossing(raw: torch.Tensor) -> torch.Tensor:
    if raw.dim() == 3:
        base = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
    elif raw.dim() == 2:
        return monotone_quantiles(raw, horizon=1, n_quantiles=raw.shape[1])
    return raw