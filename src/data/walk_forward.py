from src.data.preprocessing import Preprocessing

class WalkForwardSplitter:
    def __init__(self, seq_length: int, pred_length: int, forecast_horizon: int = 1,
                 train_ratio: float = 0.7, train_size: int | None = None,
                 val_size: int = 12, test_size: int = 24, step: int | None = None,
                 n_folds: int = 1):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.forecast_horizon = forecast_horizon
        self.val_size = val_size
        self.test_size = test_size
        self.step = step if step is not None else val_size
        self.n_folds = int(n_folds)
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least 1")

        self.data = Preprocessing().preprocess()
        self.n = len(self.data)
        self.initial_train_size = train_size if train_size is not None else int(self.n * train_ratio)
        self.min_split_size = seq_length + (forecast_horizon - 1) + pred_length
        self.validation_start = self.initial_train_size
        self.validation_end = self.validation_start + self.val_size
        self.final_test_start = self.validation_end

        expected_n = self.initial_train_size + self.val_size + self.test_size
        if expected_n != self.n:
            raise ValueError(
                "Invalid temporal split: "
                f"train ({self.initial_train_size}) + validation ({self.val_size}) + "
                f"test ({self.test_size}) = {expected_n}, but dataset has {self.n} rows."
            )

    def _has_enough_samples(self, size: int) -> bool:
        return size >= self.min_split_size

    def get_splits(self):
        first_origin = self.initial_train_size - (self.n_folds - 1) * self.step
        for fold_idx in range(self.n_folds):
            origin = first_origin + fold_idx * self.step
            val_end = origin + self.val_size
            if val_end > self.validation_end:
                raise ValueError(
                    f"Walk-forward fold {fold_idx + 1} ends at {val_end}, "
                    f"past validation boundary {self.validation_end}"
                )
            if not self._has_enough_samples(origin):
                raise ValueError(
                    f"Walk-forward fold {fold_idx + 1} has only {origin} training rows; "
                    f"at least {self.min_split_size} are required"
                )
            yield {
                "train_end": origin,
                "val_start": origin,
                "val_end": val_end,
                "fold": fold_idx + 1,
            }

    def get_final_test(self):
        return {
            "train_end": self.validation_end,
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
            val_samples = max(0, self.val_size - self.pred_length + 1)
            print(
                f"{s['fold']:<6} "
                f"[0:{s['train_end']}]{'':<4} "
                f"[{s['train_end']}:{s['val_end']}]{'':<2} "
                f"{train_samples:<15} {val_samples:<12}"
            )
        final = self.get_final_test()
        print("-" * 70)
        print(f"Final Test Window:")
        print(f"Train: [0:{final['train_end']}], Validation: [{self.validation_start}:{self.validation_end}], Test: [{final['test_start']}:{final['test_end']}], Test samples: {self.test_size-self.pred_length+1}")
