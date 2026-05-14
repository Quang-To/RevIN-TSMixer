import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import numpy as np


class TimeSeriesDecomposition(nn.Module):
    def __init__(self, seq_length: int, seasonal_period: int = 4, method: str = "ma"):
        super().__init__()
        self.seq_length = seq_length
        self.seasonal_period = seasonal_period
        self.method = method
        
        self.trend_kernel_size = seasonal_period * 2 - 1
        if self.trend_kernel_size > seq_length:
            self.trend_kernel_size = seq_length
        
    def forward(self, x: torch.Tensor) -> tuple:
        batch_size, seq_len, n_features = x.shape
        target = x[:, :, -1] 
        trend_target = self._extract_trend(target) 
        detrended = target - trend_target 
        seasonal_target = self._extract_seasonal(detrended) 
        residual_target = target - trend_target - seasonal_target
        trend = self._expand_to_features(trend_target, x[:, :, :-1], n_features)
        seasonal = self._expand_to_features(seasonal_target, x[:, :, :-1], n_features)
        residual = self._expand_to_features(residual_target, x[:, :, :-1], n_features)
        
        return trend, seasonal, residual
    
    def _extract_trend(self, target: torch.Tensor) -> torch.Tensor:
        if self.method == "ma":
            return self._moving_average_trend(target)
        elif self.method == "savgol":
            return self._savgol_trend(target)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
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
        
        self.decomposition = TimeSeriesDecomposition(seq_length, initial_period, method="ma")
    
    def forward(self, x: torch.Tensor) -> tuple:
        if self.learnable_period:
            period = max(2, min(self.seq_length // 2, int(self.seasonal_period.item())))
            self.decomposition.seasonal_period = period
            self.decomposition.trend_kernel_size = period * 2 - 1
        
        return self.decomposition(x)
