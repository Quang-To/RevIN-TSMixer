"""Oracle forecasts for calibrated simulation scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

def innovation_quantile(innov: tuple, tau: float) -> float:
    kind, val = innov
    if kind == "t":
        nu = float(val)
        return float(stats.t.ppf(tau, nu) * np.sqrt((nu - 2.0) / nu))
    return float(np.quantile(val, tau, method="inverted_cdf"))


class OracleDGP:
    """
    Oracle một bước: biết tham số DGP, quan sát quá khứ thật.
    forecast_path(meta, quantiles) -> {"mean": (n,), "P5": (n,), ...} cho mọi t.
    """
    def __init__(self, df_base: pd.DataFrame | None = None):
        pass

    @staticmethod
    def forecast_path(meta: dict, quantiles: tuple = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
        cm, sc = meta["cond_mean"], meta["scale"]
        out = {"mean": cm.copy()}
        for tau in quantiles:
            out[f"P{int(round(tau * 100))}"] = np.maximum(cm + sc * innovation_quantile(meta["innov"], tau), 0.0)
        return out


# ==============================================================================
# 4. ĐẶC TẢ KỊCH BẢN (SINH TỪ THAM SỐ HIỆU CHỈNH THẬT -> KHÔNG LỆCH VỚI CODE)
# ==============================================================================
