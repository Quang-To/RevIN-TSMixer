import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


class WindowDataset(Dataset):
    """Cửa sổ trượt đơn biến: input (lookback,) -> target (horizon,)."""

    def __init__(self, series: np.ndarray, lookback: int, horizon: int, t_range: Tuple[int, int]):
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        start = max(t_range[0], lookback)
        end = min(t_range[1], len(series) - horizon + 1)
        self.targets = list(range(start, end))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        t = self.targets[idx]
        x = self.series[t - self.lookback : t]
        y = self.series[t : t + self.horizon]
        meta = torch.tensor([t, 1.0], dtype=torch.float32)
        return x, y, meta


class MultiWindowDataset(Dataset):
    """Cửa sổ trượt đa biến: input (lookback, F) -> target (horizon,) của biến mục tiêu target_idx."""

    def __init__(self, features: np.ndarray, lookback: int, horizon: int,
                 t_range: Tuple[int, int], target_idx: int = 0):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.target_idx = target_idx
        start = max(t_range[0], lookback)
        end = min(t_range[1], len(features) - horizon + 1)
        self.targets = list(range(start, end))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        t = self.targets[idx]
        x = self.features[t - self.lookback : t]                 # (L, F)
        y = self.features[t : t + self.horizon, self.target_idx]  # (H,)
        meta = torch.tensor([t, 1.0], dtype=torch.float32)
        return x, y, meta


class TimeSeriesData(Dataset):
    """Lớp tương thích cho Trainer: tự động chọn WindowDataset hoặc MultiWindowDataset."""

    def __init__(self, seq_length: int, batch_size: int, pred_length: int = 1,
                 split_start: int = 0, split_end: Optional[int] = None,
                 forecast_horizon: int = 1, use_log_return: bool = False,
                 is_multivariate: bool = True, data_dict: Optional[dict] = None):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.forecast_horizon = forecast_horizon
        self.is_multivariate = is_multivariate

        if data_dict is None:
            from .data_loader import load_series, load_multivariate, split_indices
            y, _ = load_series()
            n = len(y)
            train_end, _ = split_indices(n)
            scale = float(y[:train_end].mean())
            ys = y / scale

            Xmv, _, _ = load_multivariate(k=6)
            col_std = Xmv[:train_end].std(axis=0)
            col_std[col_std == 0] = 1.0
            Xs = Xmv / col_std
            data_dict = {
                "ys": ys,
                "scale": scale,
                "Xs": Xs,
                "target_std": float(col_std[0]),
                "n_features": Xmv.shape[1],
            }

        self.data_dict = data_dict
        split_end_val = split_end if split_end is not None else len(data_dict["ys"])
        t_range = (split_start, split_end_val)

        if is_multivariate:
            self.inner_ds = MultiWindowDataset(data_dict["Xs"], seq_length, pred_length, t_range, target_idx=0)
            self.target_scale = data_dict["target_std"]
        else:
            self.inner_ds = WindowDataset(data_dict["ys"], seq_length, pred_length, t_range)
            self.target_scale = data_dict["scale"]

        self.loader = DataLoader(self.inner_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    def __len__(self):
        return len(self.inner_ds)

    def __getitem__(self, idx):
        return self.inner_ds[idx]

    def get_loader(self):
        return self.loader
