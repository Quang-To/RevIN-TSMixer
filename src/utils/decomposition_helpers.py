import numpy as np
import torch
import logging
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL

# Global cache for STL decomposition
# Key: (period, seasonal, trend, t_end, len_y_full)
# Value: (trend_total, seasonal_total, resid_total)
GLOBAL_DECOMP_CACHE = {}

def validate_stl_params(N, period, trend):
    """Validation checks as per section 4 of requirements."""
    warnings_list = []
    ratio = trend / N
    if ratio > 0.85:
        warnings_list.append(f"trend/N ratio {ratio:.2f} > 0.85")
    n_cycles = N / period
    if n_cycles < 3:
        warnings_list.append(f"n_cycles {n_cycles:.2f} < 3")
    return ratio, n_cycles, warnings_list

def moving_average_fallback(y, period=4):
    """Fallback moving average decomposition if STL fails."""
    N = len(y)
    trend = np.zeros(N)
    window = period * 2
    if window % 2 == 0:
        window += 1
    window = min(window, N)
    if window % 2 == 0:
        window = max(1, window - 1)
        
    pad_size = window // 2
    padded = np.pad(y, pad_size, mode='reflect')
    for i in range(N):
        trend[i] = np.mean(padded[i : i + window])
        
    detrended = y - trend
    
    seasonal = np.zeros(N)
    for p in range(period):
        indices = np.arange(p, N, period)
        if len(indices) > 0:
            val = np.mean(detrended[indices])
            seasonal[indices] = val
            
    resid = y - trend - seasonal
    return trend, seasonal, resid

def decompose_stl(y, period=4, seasonal=7, trend=41):
    """
    Decompose a single target time series using STL.
    Includes validation/sanity checks and fallback.
    """
    N = len(y)
    ratio, n_cycles, warnings_list = validate_stl_params(N, period, trend)
    
    for warn in warnings_list:
        logging.warning(warn)
        
    try:
        res = STL(y, period=period, seasonal=seasonal, trend=trend, robust=False).fit()
        return res.trend, res.seasonal, res.resid, ratio, n_cycles, warnings_list
    except Exception as e:
        logging.warning(f"STL decomposition failed: {e}. Falling back to Moving Average.")
        trend_arr, seasonal_arr, resid_arr = moving_average_fallback(y, period)
        return trend_arr, seasonal_arr, resid_arr, ratio, n_cycles, warnings_list

def extend_decomposition(y_train, trend_train, seasonal_train, horizon, period=4, trend_window=41):
    """
    Extends decomposition to Val/Test period (no leakage).
    - Seasonal: tile the last full cycle of length 'period'.
    - Trend: fit a Linear Regression on the last min(len, 2 * trend_window) points of trend.
    """
    # 1. Extend Seasonal
    last_cycle = seasonal_train[-period:]
    n_repeats = (horizon + period - 1) // period
    seasonal_extended = np.tile(last_cycle, n_repeats)[:horizon]
    
    # 2. Extend Trend
    segment_len = min(len(trend_train), 2 * trend_window)
    trend_segment = trend_train[-segment_len:]
    
    # Fit Linear Regression on this segment
    X = np.arange(segment_len).reshape(-1, 1)
    y = trend_segment
    reg = LinearRegression().fit(X, y)
    
    # Extrapolate
    X_pred = np.arange(segment_len, segment_len + horizon).reshape(-1, 1)
    trend_extended = reg.predict(X_pred)
    
    return trend_extended, seasonal_extended

def get_or_compute_decomposition(y_full, t_end, period=4, seasonal=7, trend=41):
    """
    Single source of truth for STL decomposition and extrapolation.
    Caches results to avoid redundant calculations across Optuna trials.
    """
    cache_key = (period, seasonal, trend, t_end, len(y_full))
    if cache_key in GLOBAL_DECOMP_CACHE:
        logging.debug(f"[Cache Hit] STL Decomp for N={t_end}")
        return GLOBAL_DECOMP_CACHE[cache_key]
        
    logging.info(f"[Cache Miss] Running STL Decomp for N={t_end}")
    
    y_train = y_full[0:t_end]
    horizon = len(y_full) - t_end
    
    # Decompose Train
    trend_train, seasonal_train, resid_train, ratio, n_cycles, warnings = decompose_stl(
        y_train, period=period, seasonal=seasonal, trend=trend
    )
    
    # Extend components
    trend_val, seasonal_val = extend_decomposition(
        y_train, trend_train, seasonal_train, horizon=horizon, period=period, trend_window=trend
    )
    
    # Reconstruct total pre-decomposed arrays
    trend_total = np.concatenate([trend_train, trend_val])
    seasonal_total = np.concatenate([seasonal_train, seasonal_val])
    resid_total = y_full - trend_total - seasonal_total
    
    # Log the summary metrics on train set
    var_resid = np.var(resid_train)
    var_trend_resid = np.var(trend_train + resid_train)
    var_seasonal_resid = np.var(seasonal_train + resid_train)
    var_y = np.var(y_train)
    
    Ft = max(0.0, 1.0 - var_resid / (var_trend_resid + 1e-8))
    Fs = max(0.0, 1.0 - var_resid / (var_seasonal_resid + 1e-8))
    pct_resid = (var_resid / (var_y + 1e-8)) * 100
    
    warn_str = ", ".join(warnings) if warnings else "None"
    logging.info(
        f"[STL Decomp Metrics] N={t_end} | Fs={Fs:.4f} | Ft={Ft:.4f} | %resid={pct_resid:.2f}% | "
        f"ratio={ratio:.4f} | n_cycles={n_cycles:.2f} | Warnings: {warn_str}"
    )
    
    GLOBAL_DECOMP_CACHE[cache_key] = (trend_total, seasonal_total, resid_total)
    return trend_total, seasonal_total, resid_total

def find_subsegment_index(y_full, target_seq):
    L = len(target_seq)
    for i in range(len(y_full) - L + 1):
        if np.allclose(y_full[i : i + L], target_seq, rtol=1e-5, atol=1e-5):
            return i
    return -1

def custom_stl_decompose(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_length = target.shape
    device = target.device
    
    trend = torch.zeros_like(target)
    seasonal = torch.zeros_like(target)
    residual = torch.zeros_like(target)
    
    y_full_np = self.y_full
    trend_total_np = self.trend_total
    seasonal_total_np = self.seasonal_total
    resid_total_np = self.resid_total
    
    for b in range(batch_size):
        target_np = target[b].detach().cpu().numpy()
        start_idx = find_subsegment_index(y_full_np, target_np)
        if start_idx != -1:
            trend[b] = torch.tensor(trend_total_np[start_idx : start_idx + seq_length], device=device, dtype=target.dtype)
            seasonal[b] = torch.tensor(seasonal_total_np[start_idx : start_idx + seq_length], device=device, dtype=target.dtype)
            residual[b] = torch.tensor(resid_total_np[start_idx : start_idx + seq_length], device=device, dtype=target.dtype)
        else:
            # Fallback
            trend_b = self._moving_average_trend(target[b:b+1])[0]
            seasonal_b = target[b] - trend_b
            trend[b] = trend_b
            seasonal[b] = seasonal_b
            residual[b] = target[b] - trend_b - seasonal_b
            
    return trend, seasonal, residual

def custom_decomposed_forecast_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Step 1: Decompose
    trend, seasonal, residual = self.decomposition(x)
    
    # Step 2: Combine seasonal and residual for seasonality branch
    seasonal_residual = seasonal + residual
    
    # Step 3: Forecast from seasonality branch (NHITS/NBEATS/TSMixer)
    seasonal_forecast = self.seasonality_branch(seasonal_residual)
    
    # Step 4: Get pre-computed trend forecast
    batch_size, seq_len, _ = x.shape
    device = x.device
    trend_forecast = torch.zeros(batch_size, self.pred_len, device=device, dtype=x.dtype)
    
    y_full_np = self.decomposition.y_full
    trend_total_np = self.decomposition.trend_total
    forecast_horizon = self.decomposition.forecast_horizon
    
    for b in range(batch_size):
        x_np = x[b, :, -1].detach().cpu().numpy()
        start_idx = find_subsegment_index(y_full_np, x_np)
        if start_idx != -1:
            y_start = start_idx + seq_len + forecast_horizon - 1
            y_end = y_start + self.pred_len
            trend_forecast[b] = torch.tensor(trend_total_np[y_start:y_end], device=device, dtype=x.dtype)
        else:
            trend_forecast[b] = self.trend_branch(trend[b:b+1])[0]
            
    # Step 5: Aggregate
    forecast = self.aggregation(trend_forecast, seasonal_forecast)
    return forecast

def custom_decomposed_forecast_get_component_forecasts(self, x: torch.Tensor) -> dict:
    trend, seasonal, residual = self.decomposition(x)
    seasonal_residual = seasonal + residual
    seasonal_forecast = self.seasonality_branch(seasonal_residual)
    
    batch_size, seq_len, _ = x.shape
    device = x.device
    trend_forecast = torch.zeros(batch_size, self.pred_len, device=device, dtype=x.dtype)
    
    y_full_np = self.decomposition.y_full
    trend_total_np = self.decomposition.trend_total
    forecast_horizon = self.decomposition.forecast_horizon
    
    for b in range(batch_size):
        x_np = x[b, :, -1].detach().cpu().numpy()
        start_idx = find_subsegment_index(y_full_np, x_np)
        if start_idx != -1:
            y_start = start_idx + seq_len + forecast_horizon - 1
            y_end = y_start + self.pred_len
            trend_forecast[b] = torch.tensor(trend_total_np[y_start:y_end], device=device, dtype=x.dtype)
        else:
            trend_forecast[b] = self.trend_branch(trend[b:b+1])[0]
            
    combined = self.aggregation(trend_forecast, seasonal_forecast)
    return {
        "trend": trend_forecast,
        "seasonal": seasonal_forecast,
        "combined": combined
    }
