import numpy as np
from src.data.preprocessing import Preprocessing

class WalkForwardSplitter:
    def __init__(self, seq_length: int, pred_length: int, forecast_horizon: int = 4,
                 train_ratio: float = 0.45, val_size: int = 23, test_size: int = 29, step: int = 3):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.forecast_horizon = forecast_horizon
        self.val_size = val_size
        self.test_size = test_size
        self.step = step if step is not None else pred_length  

        self.data = Preprocessing().preprocess()
        self.n = len(self.data)
        self.initial_train_size = int(self.n * train_ratio)
        self.min_split_size = seq_length + (forecast_horizon - 1) + pred_length
        self.final_test_start = self.n - self.test_size

    def _has_enough_samples(self, size: int) -> bool:
        return size >= self.min_split_size

    def get_splits(self):
        origin = self.initial_train_size
        while True:
            val_end = origin + self.val_size
            if val_end > self.final_test_start:
                break
            train_ok = self._has_enough_samples(origin)
            if train_ok:
                yield {
                    "train_end": origin,
                    "val_end": val_end,
                    "fold": (origin - self.initial_train_size) // self.step + 1,
                }
            origin += self.step

    def get_final_test(self):
        return {
            "train_end": self.final_test_start,
            "test_start": self.final_test_start,
            "test_end": self.n
        }

    def n_splits(self) -> int:
        """Tổng số folds hợp lệ."""
        return sum(1 for _ in self.get_splits())

    def summary(self):
        """In tóm tắt các folds."""
        print(f"Dataset size : {self.n}")
        print(f"Min split    : {self.min_split_size}")
        print(f"{'Fold':<6} {'Train':<12} {'Val':<12} {'Train samples':<15} {'Val samples':<12}")
        print("-" * 70)
        for s in self.get_splits():
            train_samples = s["train_end"] - self.min_split_size + 1
            val_samples = self.val_size - self.min_split_size + 1
            print(
                f"{s['fold']:<6} "
                f"[0:{s['train_end']}]{'':<4} "
                f"[{s['train_end']}:{s['val_end']}]{'':<2} "
                f"{train_samples:<15} {val_samples:<12}"
            )
        final = self.get_final_test()
        print("-" * 70)
        print(f"Final Test Window:")
        print(f"Train: [0:{final['train_end']}], Test: [{final['test_start']}:{final['test_end']}], Test samples: {final['test_end']-final['test_start']-self.min_split_size+1}")