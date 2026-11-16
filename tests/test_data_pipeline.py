import numpy as np
import pandas as pd

from src.data.data_loader import select_top_exog, split_indices


def test_split_indices_preserves_temporal_partition():
    train_end, val_end = split_indices(120)

    assert (train_end, val_end) == (84, 96)
    assert train_end < val_end < 120


def test_select_top_exog_uses_training_rows_only():
    n = 12
    train_end = 8
    target = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "Quantity": target,
            "CompetitorQuantity": np.arange(n, dtype=float),
            "PromotionAmount": np.array([4, 1, 3, 2, 8, 7, 6, 5, 1000, -1000, 1000, -1000], dtype=float),
            "Imports": np.arange(n, dtype=float) ** 2,
        }
    )

    selected = select_top_exog(df, train_end=train_end, k=1)

    assert selected == ["CompetitorQuantity"]
