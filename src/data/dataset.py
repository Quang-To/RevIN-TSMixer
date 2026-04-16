import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import Preprocessing

class TimeSeriesData(Dataset):
    def __init__(self, seq_length: int, batch_size: int, pred_length: int = 3, split_start: int = 0, split_end: int = None, forecast_horizon: int = 4):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.forecast_horizon = forecast_horizon
        data = Preprocessing().preprocess()
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.split_start = split_start
        self.split_end = split_end if split_end is not None else len(self.data)
        self.loader = DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=True, generator=torch.Generator().manual_seed(42),)

    def __len__(self):
        max_start = self.split_end - (self.seq_length + (self.forecast_horizon - 1) + self.pred_length)
        return max(0, max_start - self.split_start + 1)

    def __getitem__(self, idx):
        real_idx = self.split_start + idx
        x = self.data[real_idx : real_idx + self.seq_length]
        y_start = real_idx + self.seq_length + self.forecast_horizon - 1
        y_end = y_start + self.pred_length
        y = self.data[y_start:y_end, -1]
        # Trả về chỉ số global của sample đầu tiên trong y (có thể chọn y_start hoặc real_idx)
        return x, y, real_idx

    def get_loader(self):
        return self.loader