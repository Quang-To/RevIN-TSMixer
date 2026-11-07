"""
Module mô phỏng dữ liệu (Data Simulation) cho bài toán dự báo nhu cầu (Quantity).
Cung cấp 6 kịch bản kiểm thử (Stress testing & Concept drift testing):

  1. S1: Phân phối chuẩn tĩnh (Homoscedastic Gaussian i.i.d)
  2. S2: Dị phương sai theo tháng thật + xu hướng giảm biên độ
  3. S3: Lệch / nặng đuôi, điều khiển Skewness & Kurtosis bằng Fleishman
  4. S4: Regime shift mean + variance (Markov Switching, THỜI GIAN LƯU LIÊN TỤC >= min_regime_len)
  5. S5: Trend + Seasonality (12 & 4 tháng) + cú sốc hiếm
  6. S6: Mixture hai đỉnh, xác suất đỉnh cao điều khiển bởi PromotionAmount đã khử trend

Quy ước chung:
- Exact Moment Matching: mọi kịch bản được co giãn affine để sample mean/std = dữ liệu gốc.
  Phép co giãn y = mu + sigma * (raw - m) / s  (m, s = mean, std mẫu của raw) được LƯU vào `meta`
  để Oracle áp dụng cùng phép biến đổi -> Oracle khớp đúng DGP thực sự của dữ liệu đã sinh.
- Sau matching, giá trị âm bị clip về 0 (nhu cầu không âm). Việc này làm lệch nhẹ moment
  (chỉ xảy ra ở S2 với ~2% seed) và được báo qua RuntimeWarning + meta["n_clipped"].
- Mọi hàm simulate_scenario_k nhận thêm `meta: dict | None`; nếu truyền dict rỗng, hàm sẽ điền
  thông tin DGP cần cho OracleDGP. Kiểu trả về giữ nguyên như bản cũ.
- Dữ liệu S1, S2, S3, S5, S6 ở n=120 trùng bit-for-bit với bản cũ (cùng seed). S4 thay đổi
  vì cách sinh regime được sửa đúng thời gian lưu.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "raw_data"
BASE_DATA_PATH = RAW_DATA_DIR / "data_TSI_v2.csv"
RESULTS_DIR = ROOT / "results"

SCENARIO_KEYS = ("s1", "s2", "s3", "s4", "s5", "s6")


# ==============================================================================
# 1. TRÍCH XUẤT ĐẶC TRƯNG THỐNG KÊ NỀN TẢNG (CALIBRATION SOURCE)
# ==============================================================================

def month_index(n: int, start_month: int) -> np.ndarray:
    """Chỉ số tháng 1..12 cho n bước liên tiếp, bắt đầu từ start_month (đúng với mọi n)."""
    return (start_month - 1 + np.arange(n)) % 12 + 1


def harmonic_design(n: int) -> np.ndarray:
    """Ma trận thiết kế [1, t, sin12, cos12, sin4, cos4] cho t = 0..n-1 (đúng với mọi n)."""
    t = np.arange(n, dtype=float)
    return np.column_stack([
        np.ones(n), t,
        np.sin(2.0 * np.pi * t / 12.0), np.cos(2.0 * np.pi * t / 12.0),
        np.sin(2.0 * np.pi * t / 4.0), np.cos(2.0 * np.pi * t / 4.0),
    ])


def detrended_promo_driver(promo: np.ndarray) -> np.ndarray:
    """Khử trend tuyến tính rồi chuẩn hóa robust (median/IQR) cho PromotionAmount."""
    t = np.arange(len(promo))
    slope_p, inter_p, _, _, _ = stats.linregress(t, promo)
    promo_detrend = promo - (inter_p + slope_p * t)
    p_med = np.median(promo_detrend)
    p_iqr = np.percentile(promo_detrend, 75) - np.percentile(promo_detrend, 25) + 1e-8
    return (promo_detrend - p_med) / p_iqr


def get_base_statistics(df_base: pd.DataFrame) -> dict:
    """Trích xuất các đặc trưng kinh tế lượng từ dữ liệu thực."""
    quantity = df_base["Quantity"].to_numpy(dtype=float)
    n = len(quantity)
    t = np.arange(n)
    month_idx = pd.to_datetime(df_base["Month"]).dt.month.to_numpy()

    mean = float(np.mean(quantity))
    std = float(np.std(quantity, ddof=1))

    monthly_stds = np.array([float(quantity[month_idx == m].std(ddof=1)) for m in range(1, 13)])
    monthly_means = np.array([float(quantity[month_idx == m].mean()) for m in range(1, 13)])

    slope_q, inter_q, _, _, _ = stats.linregress(t, quantity)
    residuals = quantity - (inter_q + slope_q * t)
    slope_vol, inter_vol, _, p_vol, _ = stats.linregress(t, np.abs(residuals))

    X_harm = harmonic_design(n)
    beta_harm, _, _, _ = np.linalg.lstsq(X_harm, quantity, rcond=None)

    z_promo = detrended_promo_driver(df_base["PromotionAmount"].to_numpy(dtype=float))

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "start_month": int(month_idx[0]),
        "month_idx": month_idx,
        "monthly_stds": monthly_stds,
        "monthly_means": monthly_means,
        "slope_vol": float(slope_vol),
        "inter_vol": float(inter_vol),
        "beta_harm": beta_harm,
        "X_harm": X_harm,
        "z_promo": z_promo,
        "skewness": float(stats.skew(quantity, bias=False)),
        "kurtosis": float(stats.kurtosis(quantity, fisher=True, bias=False)),
    }


def match_moments_params(x: np.ndarray, target_mean: float, target_std: float):
    """Chuẩn hóa affine; trả về (x_matched, m, s) với m, s là mean/std mẫu của x trước khi co giãn."""
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1) + 1e-12)
    return target_mean + target_std * (x - m) / s, m, s


def match_moments(x: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Chuẩn hóa Affine để Sample Mean và Sample Std khớp chính xác với dữ liệu gốc."""
    return match_moments_params(x, target_mean, target_std)[0]


def _finish(raw, base_stats, meta, mixture=None, fleishman=None):
    """
    Matching + clip + ghi meta cho Oracle.
    mixture = (w, mu, sd), mỗi mảng shape (K, n): raw ~ sum_k w_k * N(mu_k, sd_k^2) tại từng t.
    fleishman = (a, b, c, d): raw = a + b Z + c Z^2 + d Z^3.
    """
    q, m, s = match_moments_params(raw, base_stats["mean"], base_stats["std"])
    n_clipped = int(np.sum(q < 0.0))
    if n_clipped:
        warnings.warn(f"{n_clipped} giá trị âm bị clip về 0 sau moment matching "
                      f"(mean/std mẫu sẽ lệch nhẹ so với dữ liệu gốc).", RuntimeWarning)
    if meta is not None:
        meta.update(m=m, s=s, n_clipped=n_clipped, mixture=mixture, fleishman=fleishman)
    return np.clip(q, a_min=0.0, a_max=None)


# ==============================================================================
# 2. HỆ THỐNG FLEISHMAN CHO KỊCH BẢN S3
# ==============================================================================

def fleishman_coefficients(skew: float, excess_kurt: float) -> tuple[float, float, float, float]:
    """
    Giải hệ Fleishman (1978): Y = a + b*Z + c*Z^2 + d*Z^3, Z ~ N(0,1), a = -c.
    """
    def equations(p):
        b, c, d = p
        eq1 = b**2 + 6.0*b*d + 2.0*c**2 + 15.0*d**2 - 1.0
        eq2 = 2.0*c*(b**2 + 24.0*b*d + 105.0*d**2 + 2.0) - skew
        eq3 = 24.0*(b*d + c**2*(1.0 + b**2 + 28.0*b*d) + d**2*(12.0 + 48.0*b*d + 141.0*c**2 + 225.0*d**2)) - excess_kurt
        return [eq1, eq2, eq3]

    sol = optimize.root(equations, [1.0, skew / 6.0, excess_kurt / 24.0], method="hybr")
    if not sol.success:
        sol = optimize.root(equations, [0.85, 0.15, 0.05], method="lm")
    b, c, d = sol.x
    return -c, b, c, d


# ==============================================================================
# 3. ĐỊNH NGHĨA 6 KỊCH BẢN MÔ PHỎNG
# ==============================================================================

def simulate_scenario_1(n_samples: int, base_stats: dict, seed: int = 42, meta: dict | None = None) -> np.ndarray:
    """Kịch bản 1: X_t ~ N(mu, sigma^2) i.i.d."""
    np.random.seed(seed)
    mu, sigma = base_stats["mean"], base_stats["std"]
    raw = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    mixture = (np.ones((1, n_samples)), np.full((1, n_samples), mu), np.full((1, n_samples), sigma))
    return _finish(raw, base_stats, meta, mixture=mixture)


def simulate_scenario_2(n_samples: int, base_stats: dict, seed: int = 42,
                        meta: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Kịch bản 2: Dị phương sai (std theo tháng thật) + decay biên độ tuyến tính theo thời gian.
    Đúng với mọi n_samples (chỉ số tháng suy ra từ tháng bắt đầu của dữ liệu gốc).
    """
    np.random.seed(seed)
    t = np.arange(n_samples)
    month_idx = month_index(n_samples, base_stats["start_month"])

    vol_decay = 1.0 + (base_stats["slope_vol"] / base_stats["inter_vol"]) * (t - np.mean(t))
    vol_decay = np.maximum(0.25, vol_decay)
    sigma_t = base_stats["monthly_stds"][month_idx - 1] * vol_decay

    raw = np.random.normal(loc=base_stats["mean"], scale=sigma_t)
    mixture = (np.ones((1, n_samples)), np.full((1, n_samples), base_stats["mean"]), sigma_t[None, :])
    return _finish(raw, base_stats, meta, mixture=mixture), sigma_t


def simulate_scenario_3(n_samples: int, base_stats: dict, seed: int = 42,
                        target_skew: float = 1.0, target_excess_kurt: float = 2.0,
                        meta: dict | None = None) -> np.ndarray:
    """Kịch bản 3: Lệch / nặng đuôi, điều khiển bằng Fleishman (target skew=1.0, excess kurt=2.0)."""
    np.random.seed(seed)
    a, b, c, d = fleishman_coefficients(skew=target_skew, excess_kurt=target_excess_kurt)
    z = np.random.normal(0.0, 1.0, size=n_samples)
    y = a + b * z + c * (z**2) + d * (z**3)
    return _finish(y, base_stats, meta, fleishman=(a, b, c, d))


def _sample_regimes_min_dwell(n: int, P: np.ndarray, min_dwell: int, max_tries: int = 1000) -> np.ndarray:
    """
    Semi-Markov: mỗi lần vào một regime thì ở lại ít nhất `min_dwell` tháng LIÊN TỤC,
    phần dư ~ Geometric(1 - P[k,k]) (giữ nhịp ở lại của ma trận P gốc); khi rời thì
    chuyển sang regime khác theo P đã chuẩn hóa phần ngoài đường chéo.
    Yêu cầu thêm: mỗi regime xuất hiện tổng cộng >= min_dwell tháng. (Đoạn cuối có thể bị cắt bởi n.)
    """
    off = P.copy()
    np.fill_diagonal(off, 0.0)
    off = off / off.sum(axis=1, keepdims=True)
    for _ in range(max_tries):
        regimes = np.empty(n, dtype=int)
        i = 0
        curr = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
        while i < n:
            stay = min_dwell + np.random.geometric(1.0 - P[curr, curr]) - 1
            regimes[i:i + stay] = curr
            i += stay
            curr = np.random.choice([0, 1, 2], p=off[curr])
        if np.bincount(regimes, minlength=3).min() >= min_dwell:
            return regimes
    raise RuntimeError("Không sinh được đường đi regime thỏa ràng buộc; giảm min_regime_len hoặc tăng n_samples.")


def simulate_scenario_4(n_samples: int, base_stats: dict, seed: int = 42,
                        min_regime_len: int = 18, meta: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Kịch bản 4: Regime shift mean + variance, 3 regime (nominal, TRƯỚC matching):
        Normal: Mean = 1.00*mu, Std = 0.75*sigma | Crisis: 0.75*mu, 1.40*sigma | Boom: 1.25*mu, 0.90*sigma
    - Thời gian lưu liên tục tối thiểu min_regime_len tháng cho mỗi lần vào regime.
    - Làm mượt biên bằng trung bình trượt 3 tháng (không phải linear ramp đúng nghĩa).
    - regime_stats báo cáo tham số THỰC sau co giãn (khác nominal).
    - Oracle cho S4 là oracle "biết regime": điều kiện theo mu_smooth/sd_smooth của từng tháng.
    """
    np.random.seed(seed)
    mu, sigma = base_stats["mean"], base_stats["std"]

    means = np.array([mu * 1.00, mu * 0.75, mu * 1.25])
    sigmas = np.array([sigma * 0.75, sigma * 1.40, sigma * 0.90])
    P = np.array([[0.92, 0.05, 0.03],
                  [0.08, 0.88, 0.04],
                  [0.06, 0.04, 0.90]])

    regimes = _sample_regimes_min_dwell(n_samples, P, min_regime_len)

    mu_smooth = pd.Series(means[regimes]).rolling(window=3, min_periods=1, center=True).mean().to_numpy()
    sig_smooth = pd.Series(sigmas[regimes]).rolling(window=3, min_periods=1, center=True).mean().to_numpy()

    raw = np.random.normal(loc=mu_smooth, scale=sig_smooth)
    mixture = (np.ones((1, n_samples)), mu_smooth[None, :], sig_smooth[None, :])
    q = _finish(raw, base_stats, meta, mixture=mixture)
    if meta is not None:
        meta["regimes"] = regimes

    regime_stats = {}
    for k, name in [(0, "Normal"), (1, "Crisis"), (2, "Boom")]:
        mask = (regimes == k)
        regime_stats[name] = {
            "count": int(np.sum(mask)),
            "pct": float(np.sum(mask) / n_samples),
            "mean": float(np.mean(q[mask])),
            "std": float(np.std(q[mask], ddof=1)) if np.sum(mask) > 1 else 0.0,
            "mean_ratio": float(np.mean(q[mask]) / mu),
            "std_ratio": float(np.std(q[mask], ddof=1) / sigma) if np.sum(mask) > 1 else 0.0,
        }
    return q, regime_stats


def simulate_scenario_5(n_samples: int, base_stats: dict, seed: int = 42,
                        noise_scale: float = 1.0, meta: dict | None = None) -> np.ndarray:
    """
    Kịch bản 5: Trend + Seasonality (12 & 4 tháng) + cú sốc hiếm (p=0.025; -2.0*sigma / +2.5*sigma).
    Lưu ý: noise_scale=0.90 khiến raw std ~0.93*target nên matching phóng tín hiệu ~7%.
    Đặt noise_scale=1.0 nếu muốn biên độ mùa vụ sát hồi quy gốc (sẽ đổi dữ liệu S5).
    Đúng với mọi n_samples (harmonic dựng lại theo t; trend ngoại suy tuyến tính).
    """
    np.random.seed(seed)
    t = np.arange(n_samples)
    beta = base_stats["beta_harm"]
    X_harm = harmonic_design(n_samples)

    signal = beta[1] * (t - np.mean(t)) + (X_harm[:, 2:] @ beta[2:])
    var_signal = np.var(signal, ddof=1)

    p_shock = 0.025
    shock_mask = np.random.binomial(1, p_shock, size=n_samples)
    shock_signs = np.random.choice([-2.0, 2.5], size=n_samples, p=[0.5, 0.5])
    shocks = shock_mask * shock_signs * base_stats["std"]
    var_shocks = np.var(shocks, ddof=1)

    var_noise = max(1.0, base_stats["std"]**2 - var_signal - var_shocks)
    noise_sd = np.sqrt(var_noise) * noise_scale
    noise = np.random.normal(0, noise_sd, size=n_samples)

    raw = signal + noise + shocks

    # Phân phối raw tại mỗi t: hỗn hợp 3 thành phần (không sốc / sốc âm / sốc dương)
    sg = base_stats["std"]
    w = np.vstack([np.full(n_samples, 1.0 - p_shock), np.full(n_samples, p_shock / 2), np.full(n_samples, p_shock / 2)])
    mu_k = np.vstack([signal, signal - 2.0 * sg, signal + 2.5 * sg])
    sd_k = np.full((3, n_samples), noise_sd)
    return _finish(raw, base_stats, meta, mixture=(w, mu_k, sd_k))


def simulate_scenario_6(df_base: pd.DataFrame, n_samples: int, base_stats: dict, seed: int = 42,
                        meta: dict | None = None) -> tuple[np.ndarray, float]:
    """
    Kịch bản 6: Mixture hai đỉnh, pi_high = 0.35 + 0.25*tanh(z_promo) (z_promo = PromotionAmount đã khử trend).
    mu_low = mu - 0.55*sigma (sd 0.45*sigma); mu_high = mu + 0.95*sigma (sd 0.55*sigma).
    Khi n_samples > số dòng gốc, driver tính lại từ df_base (bảng ngoại sinh đã mở rộng).
    """
    np.random.seed(seed)
    mu, sigma = base_stats["mean"], base_stats["std"]

    if n_samples <= base_stats["n"]:
        z_promo = base_stats["z_promo"][:n_samples]
    else:
        assert len(df_base) >= n_samples, "df_base phải có ít nhất n_samples dòng (dùng bảng đã mở rộng)."
        z_promo = detrended_promo_driver(df_base["PromotionAmount"].to_numpy(dtype=float)[:n_samples])

    pi_high = np.clip(0.35 + 0.25 * np.tanh(z_promo), 0.10, 0.60)

    mu_low, sigma_low = mu - 0.55 * sigma, 0.45 * sigma
    mu_high, sigma_high = mu + 0.95 * sigma, 0.55 * sigma

    is_high = np.random.rand(n_samples) < pi_high
    raw = np.where(
        is_high,
        np.random.normal(mu_high, sigma_high, size=n_samples),
        np.random.normal(mu_low, sigma_low, size=n_samples),
    )
    mixture = (
        np.vstack([pi_high, 1.0 - pi_high]),
        np.vstack([np.full(n_samples, mu_high), np.full(n_samples, mu_low)]),
        np.vstack([np.full(n_samples, sigma_high), np.full(n_samples, sigma_low)]),
    )
    q = _finish(raw, base_stats, meta, mixture=mixture)
    return q, float(np.mean(pi_high))


def extend_exogenous_features(df_base: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Mở rộng chuỗi ngoại sinh khi n_samples > len(df_base)."""
    np.random.seed(seed)
    n_base = len(df_base)
    exog_cols = [c for c in df_base.columns if c not in ("Month", "Quantity")]
    start_date = pd.to_datetime(df_base["Month"].iloc[0]) if "Month" in df_base.columns else pd.to_datetime("2015-01-01")
    months = pd.date_range(start=start_date, periods=n_samples, freq="MS").strftime("%Y-%m")
    df_ext = pd.DataFrame({"Month": months})
    repeats = int(np.ceil(n_samples / n_base))
    for col in exog_cols:
        base_vals = df_base[col].to_numpy()
        tiled = np.tile(base_vals, repeats)[:n_samples]
        noise = np.random.normal(0, 0.01 * (np.std(base_vals) + 1e-8), size=n_samples)
        df_ext[col] = tiled + noise
    return df_ext


def simulate_by_key(key: str, df_template: pd.DataFrame, n: int, base_stats: dict, seed: int = 42,
                    meta: dict | None = None):
    """Gọi kịch bản theo tên. Trả về (q, extra) với extra = sigma_t (s2) | regime_stats (s4) | mean pi_high (s6) | None."""
    if key == "s1":
        return simulate_scenario_1(n, base_stats, seed, meta=meta), None
    if key == "s2":
        return simulate_scenario_2(n, base_stats, seed, meta=meta)
    if key == "s3":
        return simulate_scenario_3(n, base_stats, seed, meta=meta), None
    if key == "s4":
        return simulate_scenario_4(n, base_stats, seed, meta=meta)
    if key == "s5":
        return simulate_scenario_5(n, base_stats, seed, meta=meta), None
    if key == "s6":
        return simulate_scenario_6(df_template, n, base_stats, seed, meta=meta)
    raise ValueError(f"Kịch bản không hợp lệ: {key}")


# ==============================================================================
# 4. ORACLE: DỰ BÁO TỐI ƯU THEO ĐÚNG DGP CỦA DỮ LIỆU ĐÃ SINH
# ==============================================================================

def _mixture_quantile(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, tau: float) -> np.ndarray:
    """Phân vị tau của hỗn hợp Gaussian tại từng t. w, mu, sd: shape (K, n). Bisection vector hóa."""
    if mu.shape[0] == 1:
        return mu[0] + sd[0] * stats.norm.ppf(tau)
    lo = np.min(mu - 12.0 * sd, axis=0)
    hi = np.max(mu + 12.0 * sd, axis=0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = np.sum(w * stats.norm.cdf((mid[None, :] - mu) / sd), axis=0)
        below = cdf < tau
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


class OracleDGP:
    """
    Dự báo tối ưu (mean + quantile) tại MỌI t, điều kiện theo thông tin mà DGP cho phép:
      S1, S3: phân phối vô điều kiện | S2: sigma theo tháng & decay | S4: biết regime (oracle "regime-known")
      S5: biết trend + mùa vụ, sốc hiếm là hỗn hợp | S6: biết pi_high của tháng đó (hỗn hợp hai đỉnh).
    Phép co giãn moment matching được áp lại qua meta["m"], meta["s"], và quantile bị clip ở 0 như dữ liệu.
    Dùng: meta = {}; y, _ = simulate_by_key("s5", df, n, stats, seed, meta=meta); oracle.forecast_path(meta)
    """
    def __init__(self, df_base: pd.DataFrame):
        self.stats = get_base_statistics(df_base)
        self.mean = self.stats["mean"]
        self.std = self.stats["std"]

    def forecast_path(self, meta: dict, quantiles: tuple = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
        m, s = meta["m"], meta["s"]

        def to_y(raw):
            return self.mean + self.std * (raw - m) / s

        out = {}
        n_len = meta.get("n", self.stats.get("n", 120))
        if meta.get("mixture") is not None:
            n_len = meta["mixture"][0].shape[1]

        if meta.get("fleishman") is not None:
            a, b, c, d = meta["fleishman"]
            mean_val = float(to_y(a + c))
            out["mean"] = np.full(n_len, mean_val)
            for tau in quantiles:
                z = stats.norm.ppf(tau)
                q_val = float(np.maximum(to_y(a + b * z + c * z**2 + d * z**3), 0.0))
                out[f"P{round(tau * 100)}"] = np.full(n_len, q_val)
        else:
            w, mu_k, sd_k = meta["mixture"]
            mean_arr = np.asarray(to_y(np.sum(w * mu_k, axis=0)))
            if mean_arr.ndim == 0:
                mean_arr = np.full(n_len, float(mean_arr))
            out["mean"] = mean_arr
            for tau in quantiles:
                q_arr = np.asarray(np.maximum(to_y(_mixture_quantile(w, mu_k, sd_k, tau)), 0.0))
                if q_arr.ndim == 0:
                    q_arr = np.full(n_len, float(q_arr))
                out[f"P{round(tau * 100)}"] = q_arr
        return out


# ==============================================================================
# 5. MONTE CARLO VALIDATION
# ==============================================================================

def run_monte_carlo_validation(df_base: pd.DataFrame, n_seeds: int = 100):
    """
    Kiểm tra độ ổn định thống kê và độ chuẩn của Oracle trên nhiều seed.
    Cover90/Cover50: tỷ lệ y nằm trong [P5,P95] / [P25,P75] của Oracle (kỳ vọng ~0.90 / ~0.50).
    Mean/Std luôn khớp tuyệt đối do exact moment matching nên chỉ để kiểm tra clip.
    """
    base_stats = get_base_statistics(df_base)
    oracle = OracleDGP(df_base)
    n = len(df_base)
    print("=" * 112)
    print(f"MONTE CARLO VALIDATION TRÊN {n_seeds} SEEDS (N={n})")
    print("=" * 112)
    print(f"{'Kịch bản':<26} | {'Mean':>10} | {'Std':>9} | {'Skewness':<13} | {'Kurtosis':<13} | {'Cover90':<7} | {'Cover50':<7} | {'Clip':<5}")
    print("-" * 112)

    for key, name in [("s1", "S1: Normal (i.i.d)"), ("s2", "S2: Heteroscedastic"), ("s3", "S3: Fleishman Skew/Kurt"),
                      ("s4", "S4: Markov Regime Shift"), ("s5", "S5: Harmonics + Shock"), ("s6", "S6: Detrended Bimodal")]:
        means, stds, skews, kurts, c90, c50, clipped = [], [], [], [], [], [], 0
        for s in range(n_seeds):
            meta = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _ = simulate_by_key(key, df_base, n, base_stats, seed=s, meta=meta)
            orc = oracle.forecast_path(meta, (0.05, 0.25, 0.75, 0.95))
            means.append(np.mean(y)); stds.append(np.std(y, ddof=1))
            skews.append(stats.skew(y)); kurts.append(stats.kurtosis(y))
            c90.append(np.mean((y >= orc["P5"]) & (y <= orc["P95"])))
            c50.append(np.mean((y >= orc["P25"]) & (y <= orc["P75"])))
            clipped += int(meta["n_clipped"] > 0)
        print(f"{name:<26} | {np.mean(means):>10,.0f} | {np.mean(stds):>9,.0f} | "
              f"{np.mean(skews):>5.2f}+/-{np.std(skews):<5.2f} | {np.mean(kurts):>5.2f}+/-{np.std(kurts):<5.2f} | "
              f"{np.mean(c90):<7.3f} | {np.mean(c50):<7.3f} | {clipped}/{n_seeds}")
    print("-" * 112)


# ==============================================================================
# 6. SINH VÀ LƯU TRỮ DỮ LIỆU CÙNG BIỂU ĐỒ
# ==============================================================================

def plot_simulation_comparison(df_base: pd.DataFrame, sim_dict: dict[str, np.ndarray], save_dir: Path):
    """Vẽ biểu đồ đối chiếu chuỗi thời gian và hàm mật độ xác suất của 6 kịch bản so với dữ liệu thực."""
    save_dir.mkdir(parents=True, exist_ok=True)
    real_q = df_base["Quantity"].to_numpy(dtype=float)
    n = len(real_q)
    t = np.arange(n)
    
    colors = {
        "Real Data": "#1f77b4",
        "S1: Homoscedastic Normal": "#2ca02c",
        "S2: Heteroscedastic": "#ff7f0e",
        "S3: Skewed & Heavy-Tail": "#d62728",
        "S4: Regime Shift": "#9467bd",
        "S5: Trend + Season + Shock": "#8c564b",
        "S6: Bimodal Mixture": "#e377c2",
    }
    
    scenario_keys = [
        ("S1: Homoscedastic Normal", "s1"),
        ("S2: Heteroscedastic", "s2"),
        ("S3: Skewed & Heavy-Tail", "s3"),
        ("S4: Regime Shift", "s4"),
        ("S5: Trend + Season + Shock", "s5"),
        ("S6: Bimodal Mixture", "s6"),
    ]
    
    # ── 1. Biểu đồ Time Series ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, (title, key) in enumerate(scenario_keys):
        ax = axes[idx]
        sim_q = sim_dict[key][:n]
        ax.plot(t, real_q, label="Real Data (Gốc)", color="gray", alpha=0.5, linestyle="--", linewidth=1.5)
        ax.plot(t, sim_q, label=title, color=colors[title], linewidth=1.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        if idx >= 4:
            ax.set_xlabel("Tháng (Timestep)", fontsize=10)
        if idx % 2 == 0:
            ax.set_ylabel("Quantity", fontsize=10)
            
    plt.tight_layout()
    time_series_path = save_dir / "simulated_scenarios_time_series.png"
    plt.savefig(time_series_path, dpi=300)
    plt.close()
    
    # ── 2. Biểu đồ Phân phối Mật độ (KDE Distribution) ─────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    x_grid = np.linspace(min(real_q.min(), 50000), max(real_q.max(), 700000), 500)
    kde_real = stats.gaussian_kde(real_q)(x_grid)
    
    for idx, (title, key) in enumerate(scenario_keys):
        ax = axes[idx]
        sim_q = sim_dict[key][:n]
        kde_sim = stats.gaussian_kde(sim_q)(x_grid)
        
        ax.plot(x_grid, kde_real, label="Real Data KDE", color="gray", linestyle="--", linewidth=2)
        ax.fill_between(x_grid, 0, kde_sim, label=f"{title} KDE", color=colors[title], alpha=0.35)
        ax.plot(x_grid, kde_sim, color=colors[title], linewidth=2)
        
        skew_val = stats.skew(sim_q)
        kurt_val = stats.kurtosis(sim_q)
        ax.text(0.05, 0.85, f"Mean: {sim_q.mean():,.0f}\nStd : {np.std(sim_q, ddof=1):,.0f}\nSkew: {skew_val:.2f}\nKurt: {kurt_val:.2f}",
                transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("Quantity", fontsize=10)
        ax.set_ylabel("Mật độ xác suất (Density)", fontsize=10)
        
    plt.tight_layout()
    dist_path = save_dir / "simulated_scenarios_distributions.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    
    print(f"[OK] Đã lưu biểu đồ Time Series : {time_series_path}")
    print(f"[OK] Đã lưu biểu đồ Distribution: {dist_path}")

def generate_and_save_simulated_datasets(df_base: pd.DataFrame, n_samples: int | None = None,
                                         seed: int = 42, plot: bool = True,
                                         save_oracle: bool = False) -> dict[str, Path]:
    """
    Tạo và lưu 6 file kịch bản mô phỏng.
    - Mặc định n_samples = len(df_base) (120 dòng); giữ nguyên Month và các cột ngoại sinh, chỉ mô phỏng Quantity.
    - n_samples khác 120: bảng ngoại sinh được mở rộng bằng extend_exogenous_features.
    - save_oracle=True: lưu thêm dự báo Oracle (mean + quantile) theo từng tháng vào results/oracle/.
    """
    n_base = len(df_base)
    n_eff = n_samples if (n_samples is not None and n_samples > 0) else n_base
    base_stats = get_base_statistics(df_base)

    if n_eff == n_base:
        df_template = df_base.copy()
    else:
        df_template = extend_exogenous_features(df_base, n_samples=n_eff, seed=seed)

    metas, extras, simulations = {}, {}, {}
    for key in SCENARIO_KEYS:
        metas[key] = {}
        simulations[key], extras[key] = simulate_by_key(key, df_template, n_eff, base_stats, seed, meta=metas[key])

    scenario_descriptions = {
        "s1": "Kịch bản 1 (Chuẩn tĩnh - Homoscedastic Gaussian)",
        "s2": "Kịch bản 2 (Dị phương sai bám sát tháng thật & decay biên độ)",
        "s3": "Kịch bản 3 (Fleishman Skewness=1.0 & Kurtosis=2.0 có kiểm soát)",
        "s4": "Kịch bản 4 (Markov Regime Shift - thời gian lưu liên tục >=18 tháng)",
        "s5": "Kịch bản 5 (Trend + Harmonics mùa 12m & 4m thật + Cú sốc hiếm)",
        "s6": "Kịch bản 6 (Bimodal Mixture điều khiển bởi Detrended PromotionAmount)",
    }

    saved_paths = {}
    print("=" * 95)
    print(f"TIẾN HÀNH TẠO 6 TẬP DỮ LIỆU MÔ PHỎNG (N={n_eff} | Target Mean={base_stats['mean']:,.2f} | Target Std={base_stats['std']:,.2f})")
    print("=" * 95)

    for key, q_vals in simulations.items():
        df_sim = df_template.copy()
        if "Quantity" in df_sim.columns:
            df_sim["Quantity"] = q_vals
        else:
            df_sim.insert(1, "Quantity", q_vals)

        path_default = RAW_DATA_DIR / f"data_simulated_{key}.csv"
        df_sim.to_csv(path_default, index=False)
        saved_paths[key] = path_default

        if n_eff != n_base:
            path_custom = RAW_DATA_DIR / f"data_simulated_{key}_{n_eff}.csv"
            df_sim.to_csv(path_custom, index=False)

        print(f"[{key.upper()}] {scenario_descriptions[key]}")
        print(f"      File  : {path_default.name}")
        print(f"      Mean  : {q_vals.mean():12.2f} | Std: {np.std(q_vals, ddof=1):12.2f} | Min: {q_vals.min():10.0f} | Max: {q_vals.max():10.0f}")
        print(f"      Skew  : {stats.skew(q_vals):6.2f} | Kurtosis: {stats.kurtosis(q_vals):6.2f}")
        if metas[key]["n_clipped"]:
            print(f"      [!] {metas[key]['n_clipped']} giá trị âm đã bị clip về 0 -> mean/std lệch nhẹ")

        if key == "s4":
            print("      Regime empirical statistics after co-giãn:")
            for reg, rst in extras[key].items():
                print(f"        * {reg:<8}: N={rst['count']:2d} ({rst['pct']:4.1%}) | Mean={rst['mean']:10.1f} ({rst['mean_ratio']:.2f} mu) | Std={rst['std']:9.1f} ({rst['std_ratio']:.2f} sigma)")
        elif key == "s6":
            print(f"      Detrended driver: pi_high trung bình = {extras[key]:.3f}")

        if save_oracle:
            orc = OracleDGP(df_base).forecast_path(metas[key])
            oracle_dir = RESULTS_DIR / "oracle"
            oracle_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(orc, index=df_template["Month"]).to_csv(oracle_dir / f"oracle_{key}.csv")
        print("-" * 95)

    if plot:
        plot_simulation_comparison(df_base, simulations, RESULTS_DIR)

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dữ liệu mô phỏng 6 kịch bản cho kiểm thử mô hình dự báo.")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Số lượng mẫu mô phỏng (mặc định: bằng số dòng data gốc = 120).")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên.")
    parser.add_argument("--no_plot", action="store_true", help="Không vẽ biểu đồ trực quan hóa.")
    parser.add_argument("--monte_carlo", action="store_true", help="Chạy kiểm chứng Monte Carlo trên 100 seeds.")
    parser.add_argument("--save_oracle", action="store_true", help="Lưu dự báo Oracle (mean + quantile) vào results/oracle/.")
    args = parser.parse_args()

    df_raw = pd.read_csv(BASE_DATA_PATH)

    if args.monte_carlo:
        run_monte_carlo_validation(df_raw, n_seeds=100)
    else:
        generate_and_save_simulated_datasets(
            df_base=df_raw,
            n_samples=args.n_samples,
            seed=args.seed,
            plot=not args.no_plot,
            save_oracle=args.save_oracle,
        )