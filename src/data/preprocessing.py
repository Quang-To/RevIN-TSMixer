from .data_loader import load_dataframe, select_top_exog, split_indices, TARGET_COL
import pandas as pd


class Preprocessing:
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.1, k: int = 6):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.k = k

    def preprocess(self):
        data = load_dataframe()
        train_end, _ = split_indices(len(data), (self.train_ratio, self.val_ratio, 1.0 - self.train_ratio - self.val_ratio))
        top = select_top_exog(data, train_end, k=self.k)
        cols = [TARGET_COL] + top
        return data[cols]
