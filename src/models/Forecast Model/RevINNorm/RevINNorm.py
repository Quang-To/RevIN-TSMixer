import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessing import Preprocessing


class TimeSeriesData(Dataset):

    def __init__(
        self,
        seq_length: int,
        batch_size: int,
        pred_length: int = 3,
        split: str = "train"
    ):

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

        data = torch.tensor(data.values, dtype=torch.float32)

        self.x_data = data[:, :-1]   
        self.y_data = data[:, -1]    

        self.loader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def __len__(self):

        return len(self.x_data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):

        x = self.x_data[
            idx: idx + self.seq_length
        ]

        y = self.y_data[
            idx + self.seq_length:
            idx + self.seq_length + self.pred_length
        ]

        return x, y

    def get_loader(self):

        return self.loader


class RevINNorm(nn.Module):

    def __init__(self, num_features=5, eps=1e-5, affine=True):

        super().__init__()

        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode="norm"):

        if mode == "norm":

            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps

            x = (x - self.mean) / self.std

            if self.affine:
                x = x * self.gamma + self.beta

            return x

        elif mode == "denorm":

            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)

            x = x * self.std + self.mean

            return x