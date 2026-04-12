import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import Preprocessing

class TimeSeriesData(Dataset):
    def __init__(self, seq_length: int, batch_size: int, pred_length: int = 3, split: str = "train", forecast_horizon: int = 4, train_ratio: float = 0.6, val_ratio: float = 0.2):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.forecast_horizon = forecast_horizon
        # Preprocessing chỉ load/clean, không split
        data = Preprocessing().preprocess()
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        if split == "train":
            start = 0
            end = train_size
            shuffle = True
        elif split == "val":
            start = train_size
            end = train_size + val_size
            shuffle = False
        elif split == "test":
            start = train_size + val_size
            end = n
            shuffle = False
        else:
            raise ValueError("split must be train/val/test")
        self.split_start = start
        self.split_end = end
        self.data = torch.tensor(data.values, dtype=torch.float32)
        generator = torch.Generator().manual_seed(42) if shuffle else None
        self.loader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def __len__(self):
        max_start = self.split_end - (self.seq_length + (self.forecast_horizon - 1) + self.pred_length)
        return max(0, max_start - self.split_start + 1)

    def __getitem__(self, idx):
        real_idx = self.split_start + idx
        x = self.data[real_idx : real_idx + self.seq_length]
        y_start = real_idx + self.seq_length + self.forecast_horizon - 1
        y_end = y_start + self.pred_length
        y = self.data[y_start:y_end, -1]
        return x, y

    def get_loader(self):
        return self.loader