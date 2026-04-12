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
        return data