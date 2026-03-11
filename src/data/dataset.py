import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import Preprocessing
from src.utils.seed import set_seed

class TimeSeriesData(Dataset):
    def __init__(self, seq_length: int, batch_size: int, pred_length: int = 3, split: str = "train"):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.batch_size = batch_size
        train, val, test = Preprocessing().preprocess()

        if split == "train":
            data = train
            shuffle = True
        elif split == "val":
            data = val
            shuffle = False
        elif split == "test":
            data = test
            shuffle = False
        else:
            raise ValueError("split must be train/val/test")
        self.data = torch.tensor(data.values, dtype=torch.float32)
        generator = torch.Generator().manual_seed(42) if shuffle else None
        self.loader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def __len__(self):
        return max(0, len(self.data) - self.seq_length - self.pred_length + 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[
            idx + self.seq_length :
            idx + self.seq_length + self.pred_length,
            -1
        ]
        return x, y

    def get_loader(self):
        return self.loader