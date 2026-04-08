from .data_loader import data_loader
import pandas as pd

class Preprocessing:
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def preprocess(self):
        data = data_loader()

        data["Month"] = pd.to_datetime(data["Month"], format="%Y-%m")
        data = data.sort_values("Month").reset_index(drop=True)
        data = data[["Imports", "IPI", "DisbursedFDI", "CompetitorQuantity", "PromotionAmount", "Quantity"]]

        n = len(data)
        train_size = int(n * self.train_ratio)
        val_size = int(n * self.val_ratio)

        train = data.iloc[:train_size]
        val = data.iloc[train_size:train_size + val_size]
        test = data.iloc[train_size + val_size:]

        return train, val, test