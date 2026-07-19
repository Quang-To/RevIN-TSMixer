import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import numpy as np
from statsmodels.tsa.seasonal import STL


class TimeSeriesDecomposition(nn.Module):
    def __init__(
        self,
        seq_length: int,
        seasonal_period: int = 4,
        method: str = "stl",
        stl_robust: bool = True,
        stl_seasonal: int = 7,
        stl_trend: int | None = None,
        stl_low_pass: int | None = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.seasonal_period = seasonal_period
        self.method = method
        self.stl_robust = bool(stl_robust)
        self.stl_seasonal = int(stl_seasonal)
        self.stl_trend = stl_trend
        self.stl_low_pass = stl_low_pass

        self.trend_kernel_size = seasonal_period * 2 - 1
        if self.trend_kernel_size > seq_length:
            self.trend_kernel_size = seq_length
        self._cache = {}
        
    def forward(self, x: torch.Tensor) -> tuple:
        if self.method == "stl":
            # Convert target values to a bytes key for caching
            target_key = x[:, :, -1].detach().cpu().numpy().tobytes()
            if target_key in self._cache:
                trend, seasonal, residual = self._cache[target_key]
                return trend.to(x.device), seasonal.to(x.device), residual.to(x.device)

        batch_size, seq_len, n_features = x.shape
        target = x[:, :, -1]

        if self.method == "stl":
            # STL is fit per sample-window only (no global fit on full series),
            # which prevents train/test leakage across folds.
            trend_target, seasonal_target, residual_target = self._stl_decompose(target)
        else:
            trend_target = self._extract_trend(target)
            detrended = target - trend_target
            seasonal_target = self._extract_seasonal(detrended)
            residual_target = target - trend_target - seasonal_target

        trend = self._expand_to_features(trend_target, x[:, :, :-1], n_features)
        seasonal = self._expand_to_features(seasonal_target, x[:, :, :-1], n_features)
        residual = self._expand_to_features(residual_target, x[:, :, :-1], n_features)
        
        if self.method == "stl":
            self._cache[target_key] = (trend.detach(), seasonal.detach(), residual.detach())
            
        return trend, seasonal, residual
    
    def _extract_trend(self, target: torch.Tensor) -> torch.Tensor:
        if self.method == "ma":
            return self._moving_average_trend(target)
        elif self.method == "savgol":
            return self._savgol_trend(target)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _stl_decompose(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit STL on each sample-window independently to avoid leakage."""
        batch_size, seq_length = target.shape
        device = target.device

        trend = torch.zeros_like(target)
        seasonal = torch.zeros_like(target)
        residual = torch.zeros_like(target)

        period = max(2, min(int(self.seasonal_period), max(2, seq_length - 1)))

        def _odd_or_none(v):
            if v is None:
                return None
            vv = int(v)
            if vv % 2 == 0:
                vv += 1
            return vv

        stl_seasonal = max(3, _odd_or_none(self.stl_seasonal) or 7)
        stl_trend = _odd_or_none(self.stl_trend)
        stl_low_pass = _odd_or_none(self.stl_low_pass)

        for b in range(batch_size):
            try:
                series_np = target[b].detach().cpu().numpy()
                stl_fit = STL(
                    series_np,
                    period=period,
                    seasonal=stl_seasonal,
                    trend=stl_trend,
                    low_pass=stl_low_pass,
                    robust=self.stl_robust,
                ).fit()
                trend[b] = torch.tensor(stl_fit.trend, device=device, dtype=target.dtype)
                seasonal[b] = torch.tensor(stl_fit.seasonal, device=device, dtype=target.dtype)
                residual[b] = torch.tensor(stl_fit.resid, device=device, dtype=target.dtype)
            except Exception:
                # Conservative fallback keeps behavior stable if STL fails.
                trend_b = self._moving_average_trend(target[b:b+1])[0]
                seasonal_b = target[b] - trend_b
                trend[b] = trend_b
                seasonal[b] = seasonal_b
                residual[b] = target[b] - trend_b - seasonal_b

        return trend, seasonal, residual
    
    def _moving_average_trend(self, target: torch.Tensor) -> torch.Tensor:
        """Extract trend using centered moving average"""
        batch_size, seq_length = target.shape
        device = target.device
        kernel_size = min(self.trend_kernel_size, seq_length)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        
        pad_size = kernel_size // 2
        padded = nn.functional.pad(target, (pad_size, pad_size), mode='reflect')
        trend = torch.zeros_like(target)
        for i in range(seq_length):
            trend[:, i] = padded[:, i:i+kernel_size].mean(dim=1)
        
        return trend
    
    def _savgol_trend(self, target: torch.Tensor) -> torch.Tensor:
        """Extract trend using Savitzky-Golay filter"""
        batch_size, seq_length = target.shape
        device = target.device
        
        kernel_size = min(self.trend_kernel_size, seq_length)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        polyorder = min(3, kernel_size - 1)
        
        trend = torch.zeros_like(target)
        for b in range(batch_size):
            try:
                trend_np = torch.tensor(
                    savgol_filter(target[b].cpu().numpy(), kernel_size, polyorder),
                    device=device,
                    dtype=target.dtype
                )
                trend[b] = trend_np
            except:
                trend[b] = self._moving_average_trend(target[b:b+1])[0]
        
        return trend
    
    def _extract_seasonal(self, detrended: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = detrended.shape
        device = detrended.device
        
        seasonal = torch.zeros_like(detrended)
        for i in range(seq_length):
            cycle_idx = i % self.seasonal_period
            positions = [cycle_idx + k * self.seasonal_period for k in range(seq_length // self.seasonal_period + 1)]
            positions = [p for p in positions if p < seq_length]
            
            if len(positions) > 0:
                seasonal[:, i] = detrended[:, positions].mean(dim=1)
        
        return seasonal
    
    def _expand_to_features(self, component: torch.Tensor, other_features: torch.Tensor, n_features: int) -> torch.Tensor:
        # Keep decomposition branches univariate (target-only) so downstream
        # models and trainers remain shape-consistent across datasets.
        return component.unsqueeze(-1)


class AdaptiveDecomposition(nn.Module):
    def __init__(self, seq_length: int, initial_period: int = 4, learnable_period: bool = False):
        super().__init__()
        self.seq_length = seq_length
        self.learnable_period = learnable_period
        
        if learnable_period:
            self.seasonal_period = nn.Parameter(torch.tensor(float(initial_period)))
        else:
            self.seasonal_period = initial_period
        
        self.decomposition = TimeSeriesDecomposition(seq_length, initial_period, method="stl")
    
    def forward(self, x: torch.Tensor) -> tuple:
        if self.learnable_period:
            period = max(2, min(self.seq_length // 2, int(self.seasonal_period.item())))
            self.decomposition.seasonal_period = period
            self.decomposition.trend_kernel_size = period * 2 - 1
        
        return self.decomposition(x)
