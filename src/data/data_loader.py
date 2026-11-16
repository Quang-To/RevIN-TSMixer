"""Pipeline dữ liệu: đọc CSV, chia train/val/test theo thời gian, chọn đặc trưng ngoại sinh."""
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "raw_data"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "data_TSI_v2.csv"

TARGET_COL = "Quantity"
EXOG_COLS = [
    "CompetitorQuantity", "PromotionAmount", "Construction", "CPI", "Exports",
    "Imports", "IPI", "RegisteredFDI", "DisbursedFDI", "RetailSales",
]


def resolve_data_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Xác định đường dẫn file dữ liệu CSV."""
    if path is None:
        try:
            import config
            filename = getattr(config, "DATA_FILE", "data_TSI_v2.csv")
            return RAW_DATA_DIR / filename if not Path(filename).is_absolute() else Path(filename)
        except Exception:
            return DEFAULT_DATA_PATH
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    if (RAW_DATA_DIR / p).exists():
        return RAW_DATA_DIR / p
    return p


def load_dataframe(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    resolved_path = resolve_data_path(path)
    df = pd.read_csv(resolved_path)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df.sort_values("Month").reset_index(drop=True)
    return df


def load_series(path: Optional[Union[str, Path]] = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Trả về chuỗi mục tiêu Quantity và trục thời gian."""
    df = load_dataframe(path)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    months = pd.DatetimeIndex(df["Month"])
    return y, months


def split_indices(n: int, ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple[int, int]:
    """Chia 7/1/2 theo thời gian.
    Trả về (train_end, val_end): train [0, train_end), val [train_end, val_end), test [val_end, n)."""
    train_end = int(round(n * ratios[0]))
    val_end = train_end + int(round(n * ratios[1]))
    return train_end, val_end


def select_top_exog(df: pd.DataFrame, train_end: int, k: int = 6) -> list[str]:
    """Chọn k biến ngoại sinh có |tương quan Pearson| với Quantity cao nhất trên tập train."""
    tr = df.iloc[:train_end]
    corrs = {c: abs(float(tr[c].corr(tr[TARGET_COL]))) for c in EXOG_COLS if c in tr.columns}
    return sorted(corrs, key=corrs.get, reverse=True)[:k]


def select_top_exog_rf(df: pd.DataFrame, train_end: int, k: int = 5) -> list[str]:
    """Chọn k biến ngoại sinh theo độ quan trọng phi tuyến (RandomForest + GradientBoosting) trên tập train."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    tr = df.iloc[:train_end]
    available_exog = [c for c in EXOG_COLS if c in tr.columns]
    X = tr[available_exog].to_numpy(dtype=float)
    y = tr[TARGET_COL].to_numpy(dtype=float)
    imp = np.zeros(len(available_exog))
    for Model in (RandomForestRegressor, GradientBoostingRegressor):
        m = Model(random_state=0).fit(X, y)
        fi = m.feature_importances_
        imp += fi / (fi.sum() + 1e-12)
    order = np.argsort(imp)[::-1]
    return [available_exog[i] for i in order[:k]]


def load_multivariate(path: Optional[Union[str, Path]] = None, k: int = 6, method: str = "corr"
                      ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Trả về ma trận đặc trưng (n, F) với cột 0 = Quantity và k biến ngoại sinh."""
    df = load_dataframe(path)
    train_end, _ = split_indices(len(df))
    top = (select_top_exog_rf(df, train_end, k) if method == "rf"
           else select_top_exog(df, train_end, k))
    cols = [TARGET_COL] + top
    X = df[cols].to_numpy(dtype=np.float64)
    months = pd.DatetimeIndex(df["Month"])
    return X, top, months


def data_loader(filename: str = "data_TSI_v2.csv") -> pd.DataFrame:
    return load_dataframe(filename)