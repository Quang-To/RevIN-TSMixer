"""Monte Carlo validation for simulated demand scenarios."""

import warnings

import numpy as np
import pandas as pd

from .calibration import get_base_statistics
from .oracle import OracleDGP
from .scenarios import SCENARIO_KEYS, SCENARIO_NAMES, simulate_by_key

def _acf(y: np.ndarray, lag: int) -> float:
    return float(pd.Series(y).autocorr(lag))


def _linear_ar_test_mae(y: np.ndarray, p: int = 24, test_start: int = 96) -> float:
    """MAE một bước của AR(p) tuyến tính fit trên y[:test_start] -> đo 'độ dễ dự báo' tuyến tính."""
    X = np.array([np.r_[1.0, y[k - p:k]] for k in range(p, len(y))])
    T = y[p:]
    ntr = test_start - p
    b = np.linalg.lstsq(X[:ntr], T[:ntr], rcond=None)[0]
    return float(np.mean(np.abs(X[ntr:] @ b - T[ntr:])))


def run_monte_carlo_validation(df_base: pd.DataFrame, n_seeds: int = 100, base_stats: dict | None = None, **overrides):
    bs = base_stats or get_base_statistics(df_base)
    n = len(df_base)
    y_real = df_base["Quantity"].to_numpy(dtype=float)
    ts = slice(int(0.8 * n), n)

    hdr = (f"{'Kịch bản':<44} | {'Mean':>8} | {'Std':>7} | {'ACF1':>5} | {'ACF12':>5} | {'ACF24':>5} | "
           f"{'AR24-MAE':>8} | {'Orc-MAE':>7} | {'Cov90':>5} | {'Cov50':>5} | Clip")
    print("=" * len(hdr))
    print(f"MONTE CARLO VALIDATION ({n_seeds} seeds, N={n}) | P={bs['season_period']} K={bs['n_harmonics']} "
          f"p={bs['ar_order']} tau={bs['regime_break']} sigma_pre={bs['sigma_pre']:,.0f} "
          f"sigma_post={bs['sigma_post']:,.0f} R2_total={bs['r2_total']:.2f}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    print(f"{'REAL data_TSI_v2':<44} | {y_real.mean():>8,.0f} | {y_real.std(ddof=1):>7,.0f} | "
          f"{_acf(y_real, 1):>5.2f} | {_acf(y_real, 12):>5.2f} | {_acf(y_real, 24):>5.2f} | "
          f"{_linear_ar_test_mae(y_real):>8,.0f} | {'-':>7} | {'-':>5} | {'-':>5} | -")

    rows = {}
    for key in SCENARIO_KEYS:
        acc = {c: [] for c in ("mean", "std", "a1", "a12", "a24", "ar", "orc", "c90", "c50")}
        clipped = 0
        for s in range(n_seeds):
            meta = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _ = simulate_by_key(key, df_base, n, bs, seed=s, meta=meta, **overrides)
            orc = OracleDGP.forecast_path(meta, (0.05, 0.25, 0.50, 0.75, 0.95))
            acc["mean"].append(y.mean()); acc["std"].append(y.std(ddof=1))
            acc["a1"].append(_acf(y, 1)); acc["a12"].append(_acf(y, 12)); acc["a24"].append(_acf(y, 24))
            acc["ar"].append(_linear_ar_test_mae(y))
            acc["orc"].append(np.mean(np.abs(orc["P50"][ts] - y[ts])))
            acc["c90"].append(np.mean((y >= orc["P5"]) & (y <= orc["P95"])))
            acc["c50"].append(np.mean((y >= orc["P25"]) & (y <= orc["P75"])))
            clipped += int(meta["n_clipped"] > 0)
        r = {c: float(np.mean(v)) for c, v in acc.items()}
        rows[key] = r
        print(f"{SCENARIO_NAMES[key]:<44} | {r['mean']:>8,.0f} | {r['std']:>7,.0f} | {r['a1']:>5.2f} | "
              f"{r['a12']:>5.2f} | {r['a24']:>5.2f} | {r['ar']:>8,.0f} | {r['orc']:>7,.0f} | "
              f"{r['c90']:>5.3f} | {r['c50']:>5.3f} | {clipped}/{n_seeds}")
    print("-" * len(hdr))
    print("AR24-MAE: MAE test của AR(24) tuyến tính fit trên 96 tháng đầu (độ dễ dự báo tuyến tính).")
    print("Orc-MAE : MAE test của median Oracle (cận dưới). Cov90/Cov50 kỳ vọng ~0.90/~0.50.")
    return rows


# ==============================================================================
# 6. SINH, LƯU TRỮ VÀ TRỰC QUAN HÓA
# ==============================================================================
