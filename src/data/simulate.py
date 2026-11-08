"""
Mô phỏng dữ liệu nhu cầu (Quantity) HIỆU CHỈNH THEO ĐỘNG HỌC của chuỗi thật (data_TSI_v2.csv).

Thiết kế chung (mọi kịch bản dùng chung một DGP nền, chỉ thêm MỘT yếu tố gây nhiễu):

    y_t = max(0, m_t + u_t)
    m_t = a + b * min(t, n_base-1) + sum_k [c_k sin(2 pi k t / P) + d_k cos(2 pi k t / P)]   (+ stressor)
    u_t = sum_{j=1..p} phi_j u_{t-j} + k * sigma_t * s_t * eps_t
    sigma_t = sigma_pre (t < tau) | sigma_post (t >= tau)

  - tau: điểm đổi chế độ nhiễu, ước lượng từ sai phân mùa vụ y_t - y_{t-P} (chuỗi thật ồn mạnh trước tau, gần tuần hoàn sau tau).
  - (a, b, c_k, d_k): hồi quy có trọng số 1/sigma^2 theo chế độ trên trend + Fourier chu kỳ P (mặc định 24, K chọn theo BIC).
  - phi: AR(p) fit bằng OLS riêng cho từng chế độ trên phần dư đã chuẩn hóa (p chọn theo BIC trên mẫu chung, ép dừng nếu cần).
  - eps_t: residual bootstrap (rút i.i.d từ phần dư AR đã chuẩn hóa theo chế độ) -> giữ hình dạng nhiễu thật.
  - k = noise_scale: núm điều chỉnh tỷ lệ tín hiệu/nhiễu (SNR) cho thí nghiệm quét.

Sáu kịch bản (s1 = nền, s2..s6 = nền + một stressor):
  S1 Base            : DGP hiệu chỉnh, nhiễu bootstrap đồng phương sai.
  S2 Heteroscedastic : s_t = h(month)^power, h ước lượng từ RMS phần dư theo tháng (khuếch đại bằng power).
  S3 Heavy-tailed    : eps_t ~ Student-t(nu) chuẩn hóa phương sai 1 thay cho bootstrap.
  S4 Level shift     : m_t += delta * 1[t >= tau], tau rơi vào giai đoạn TEST, |delta| ~ U(15%, 25%) * mean.
  S5 Trend break     : m_t += Delta_b * max(0, t - tau), tau quanh cuối Train/Val, độ dốc mới đổi dấu ngẫu nhiên.
  S6 Exogenous driver: m_t += beta * k * sigma_t * g(z_{t-lag}), z = PromotionAmount khử trend (robust), g = tanh chuẩn hóa.
                       Hiệu ứng đi qua giá trị TRỄ của biến ngoại sinh nên mô hình đa biến quan sát được nó.

Oracle (OracleDGP): biết DGP & tham số, dự báo một bước với quá khứ thật:
    E[y_t | F_{t-1}] = m_t + sum phi_j u_{t-j};   Q_tau = max(0, E + k sigma s_t F_eps^{-1}(tau)).
  Đây là cận dưới lý thuyết cho pinball loss một bước (S4/S5 là oracle "biết điểm gãy").

Common random numbers: luồng nhiễu (rng_noise) tách khỏi luồng tham số kịch bản (rng_scn), nên cùng seed thì
S1/S2/S4/S5/S6 dùng chung chuỗi nhiễu -> so sánh giữa các kịch bản chỉ khác ở stressor.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "raw_data"
BASE_DATA_PATH = RAW_DATA_DIR / "data_TSI_v2.csv"
RESULTS_DIR = ROOT / "results"

SCENARIO_KEYS = ("s1", "s2", "s3", "s4", "s5", "s6")

SCENARIO_NAMES = {
    "s1": "S1: Calibrated Base (Trend + Seasonal + AR, bootstrap noise)",
    "s2": "S2: Base + Seasonal Heteroscedasticity",
    "s3": "S3: Base + Heavy-tailed Innovations (Student-t)",
    "s4": "S4: Base + Level Shift in Test Period",
    "s5": "S5: Base + Trend Break",
    "s6": "S6: Base + Lagged Exogenous Driver (Promotion)",
}

# Tham số stressor mặc định. Mọi giá trị đều override được qua simulate_by_key(..., **overrides).
DEFAULT_SCENARIO_PARAMS = {
    "noise_scale": 1.0,            # k: nhân với sigma của innovation (SNR sweep)
    "hetero_power": 1.0,           # S2: 1.0 = đúng pattern RMS phần dư theo tháng của dữ liệu thật (>1 để khuếch đại)
    "t_df": 3.0,                   # S3: bậc tự do Student-t
    "shift_frac": (0.15, 0.25),    # S4: |delta| / mean
    "shift_window": (0.80, 0.90),  # S4: vị trí điểm gãy theo tỷ lệ chiều dài (0.8n..0.9n = đầu giai đoạn test)
    "slope_frac": (0.004, 0.008),  # S5: |thay đổi độ dốc| / mean mỗi tháng
    "break_window": (0.70, 0.80),  # S5: vị trí gãy trend (cuối Train/Val)
    "exog_beta": 1.0,              # S6: cường độ driver tính theo đơn vị k*sigma
    "exog_lag": 1,                 # S6: độ trễ của driver
}

AR_STABILITY_LIMIT = 0.97
BURN_IN = 120


# ==============================================================================
# 1. HIỆU CHỈNH (CALIBRATION) TỪ DỮ LIỆU THẬT
# ==============================================================================

def month_index(n: int, start_month: int) -> np.ndarray:
    """Chỉ số tháng 1..12 cho n bước liên tiếp, bắt đầu từ start_month."""
    return (start_month - 1 + np.arange(n)) % 12 + 1


def fourier_design(t: np.ndarray, period: int, n_harmonics: int, trend_cap: int | None = None) -> np.ndarray:
    """Ma trận [1, t, sin/cos k=1..K]. Trend bị chặn tại trend_cap để chuỗi dài không trôi về âm."""
    t = np.asarray(t, dtype=float)
    t_trend = np.minimum(t, trend_cap) if trend_cap is not None else t
    cols = [np.ones_like(t), t_trend]
    for k in range(1, n_harmonics + 1):
        if 2 * k != period:                       # sin tại tần số Nyquist luôn bằng 0
            cols.append(np.sin(2.0 * np.pi * k * t / period))
        cols.append(np.cos(2.0 * np.pi * k * t / period))
    return np.column_stack(cols)


def detrended_promo_driver(promo: np.ndarray) -> np.ndarray:
    """Khử trend tuyến tính rồi chuẩn hóa robust (median/IQR) cho PromotionAmount."""
    t = np.arange(len(promo))
    slope_p, inter_p, _, _, _ = stats.linregress(t, promo)
    promo_detrend = promo - (inter_p + slope_p * t)
    p_med = np.median(promo_detrend)
    p_iqr = np.percentile(promo_detrend, 75) - np.percentile(promo_detrend, 25) + 1e-8
    return (promo_detrend - p_med) / p_iqr


def _ar_design(u: np.ndarray, p: int, start: int):
    rows = range(start, len(u))
    X = np.array([u[k - p:k][::-1] for k in rows]) if p > 0 else np.zeros((len(rows), 0))
    return X, u[start:]


def _spectral_radius(phi: np.ndarray) -> float:
    p = len(phi)
    if p == 0:
        return 0.0
    comp = np.zeros((p, p))
    comp[0] = phi
    comp[1:, :-1] = np.eye(p - 1)
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


def _select_ar(u: np.ndarray, max_p: int):
    """Chọn bậc AR theo BIC trên mẫu chung (bắt đầu tại max_p) rồi fit lại trên toàn bộ mẫu khả dụng."""
    best_p, best_bic = 0, np.inf
    for p in range(0, max_p + 1):
        X, T = _ar_design(u, p, max_p)
        phi = np.linalg.lstsq(X, T, rcond=None)[0] if p > 0 else np.zeros(0)
        res = T - X @ phi
        m = len(res)
        bic = m * np.log(res @ res / m) + p * np.log(m)
        if bic < best_bic:
            best_p, best_bic = p, bic
    X, T = _ar_design(u, best_p, best_p)
    phi = np.linalg.lstsq(X, T, rcond=None)[0] if best_p > 0 else np.zeros(0)
    res = T - X @ phi

    rho = _spectral_radius(phi)
    if rho >= AR_STABILITY_LIMIT:                 # ép dừng: phi_j *= c^j tương đương co nghiệm theo hệ số c
        c = AR_STABILITY_LIMIT / rho
        phi = phi * c ** np.arange(1, best_p + 1)
        res = T - X @ phi
    return best_p, phi, res, rho


def parse_regime_break(value: str | int | None) -> str | int | None:
    """Chuyển tham số dòng lệnh thành 'auto' | None | t (số nguyên)."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip().lower()
    return text if text == "auto" else None if text == "none" else int(text)


def detect_regime_break(y: np.ndarray, period: int) -> int | None:
    """
    Ước lượng điểm đổi chế độ tau bằng change-point phương sai trên d_i = y_{i+P} - y_i (cực đại likelihood Gauss).
    Dùng sai phân mùa vụ để không phải fit Fourier trên đoạn ngắn. d_i nhiễu mạnh khi y_i hoặc y_{i+P} thuộc chế độ ồn,
    nên: ồn -> êm thì tau = k; êm -> ồn thì tau = k + P. Trả về None nếu chuỗi quá ngắn.
    """
    d = y[period:] - y[:-period]
    m = len(d)
    if m < 2 * period:
        return None
    best_cost, best_k = np.inf, None
    for k in range(period, m - period + 1):
        cost = k * np.log(np.mean(d[:k] ** 2) + 1e-12) + (m - k) * np.log(np.mean(d[k:] ** 2) + 1e-12)
        if cost < best_cost:
            best_cost, best_k = cost, k
    noisy_first = np.mean(d[:best_k] ** 2) >= np.mean(d[best_k:] ** 2)
    return int(best_k if noisy_first else best_k + period)


def _fit_mean(X: np.ndarray, y: np.ndarray, tau: int | None, n_iter: int = 25):
    """OLS (tau=None) hoặc WLS lặp với trọng số 1/sigma theo chế độ. Trả về (beta, s_pre, s_post, -2logL rút gọn)."""
    n = len(y)
    if tau is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        s = float(np.sqrt(np.mean((y - X @ beta) ** 2)))
        return beta, s, s, n * np.log(s ** 2)
    is_pre = np.arange(n) < tau
    w = np.ones(n)
    for _ in range(n_iter):
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        r = y - X @ beta
        s_pre = float(np.sqrt(np.mean(r[is_pre] ** 2)))
        s_post = float(np.sqrt(np.mean(r[~is_pre] ** 2)))
        w = np.where(is_pre, 1.0 / s_pre, 1.0 / s_post)
    return beta, s_pre, s_post, tau * np.log(s_pre ** 2) + (n - tau) * np.log(s_post ** 2)


def sigma_path(n: int, base_stats: dict) -> np.ndarray:
    """sigma_t của innovation: sigma_pre trước tau, sigma_post từ tau trở đi (hằng số nếu một chế độ)."""
    tau = base_stats["regime_break"]
    out = np.full(n, base_stats["sigma_post"], dtype=float)
    if tau is not None:
        out[:tau] = base_stats["sigma_pre"]
    return out


def get_base_statistics(df_base: pd.DataFrame, season_period: int = 24, max_ar_order: int = 6,
                        calib_end: int | None = None, regime_break: str | int | None = "auto") -> dict:
    """
    Hiệu chỉnh DGP nền từ dữ liệu thật.
    - season_period: chu kỳ mùa vụ P (data_TSI_v2 có tự tương quan lag-24 ~0.70, lag-12 âm).
    - calib_end: chỉ dùng y[:calib_end] để hiệu chỉnh (None = toàn bộ). Dùng khi muốn loại đoạn dữ liệu nghi vấn.
    - regime_break: 'auto' (ước lượng tau), t cố định, hoặc None (một chế độ nhiễu như thiết kế cũ).
    """
    y_all = df_base["Quantity"].to_numpy(dtype=float)
    n_all = len(y_all)
    n_c = n_all if calib_end is None else int(calib_end)
    y = y_all[:n_c]
    t = np.arange(n_c)
    month_idx = pd.to_datetime(df_base["Month"]).dt.month.to_numpy()
    tau = detect_regime_break(y, season_period) if regime_break == "auto" else regime_break

    # (a) Trend + Fourier: chọn K theo BIC
    best = None
    for K in range(0, season_period // 2 + 1):
        X = fourier_design(t, season_period, K)
        beta, _, _, neg2ll = _fit_mean(X, y, tau)
        bic = neg2ll + X.shape[1] * np.log(n_c)
        if best is None or bic < best[0]:
            best = (bic, K, beta)
    _, K, beta = best
    X = fourier_design(t, season_period, K)
    beta, s_pre, s_post, _ = _fit_mean(X, y, tau)
    u = y - X @ beta
    is_pre = np.zeros(n_c, dtype=bool) if tau is None else t < tau
    u_scale = np.where(is_pre, s_pre, s_post)

    # (b) AR(p) fit riêng từng chế độ: tự tương quan của đoạn êm không được khuếch đại đoạn ồn
    segments = [(0, n_c)] if tau is None else [(0, tau), (tau, n_c)]
    ar_fits = [_select_ar(u[a:b] / u_scale[a], max_ar_order) for a, b in segments]
    sigmas, units, t_res = [], [], []
    for (a, b), (p_i, _, res_i, _) in zip(segments, ar_fits):
        m_i = len(res_i)
        sigmas.append(float(np.std(res_i, ddof=1) * u_scale[a] * np.sqrt((m_i - 1) / max(m_i - p_i - 1, 1))))
        units.append(res_i / res_i.std(ddof=0))
        t_res.append(np.arange(a + p_i, b))
    p, phi, _, rho_raw = ar_fits[-1]
    phi_pre = ar_fits[0][1]
    sigma_pre, sigma_post = sigmas[0], sigmas[-1]
    sigma = float(np.sqrt(sum(len(x) * s ** 2 for x, s in zip(units, sigmas)) / sum(len(x) for x in units)))
    res_unit = np.concatenate(units)
    t_res = np.concatenate(t_res)
    pool = (res_unit - res_unit.mean()) / res_unit.std(ddof=0)

    # (c) Pattern phương sai theo tháng (cho S2) trên phần dư đã khử chế độ, chuẩn hóa mean(h^2) = 1
    res_months = month_idx[:n_c][t_res]
    rms =np.array([np.sqrt(np.mean(res_unit[res_months == mth] ** 2)) if np.any(res_months == mth) else np.nan
                    for mth in range(1, 13)])
    rms = np.where(np.isnan(rms), np.nanmean(rms), rms)
    monthly_scale = rms / np.sqrt(np.mean(rms ** 2))

    var_y = float(np.var(y, ddof=1))
    return {
        "n": n_all,
        "calib_n": n_c,
        "mean": float(np.mean(y_all)),
        "std": float(np.std(y_all, ddof=1)),
        "start_month": int(month_idx[0]),
        "month_idx": month_idx,
        "season_period": int(season_period),
        "n_harmonics": int(K),
        "beta": beta,
        "trend_slope": float(beta[1]),
        "ar_order": int(p),
        "phi": phi,
        "phi_pre": phi_pre,
        "ar_spectral_radius_raw": rho_raw,
        "ar_spectral_radius": _spectral_radius(phi),
        "sigma": sigma,
        "regime_break": None if tau is None else int(tau),
        "sigma_pre": sigma_pre,
        "sigma_post": sigma_post,
        "innov_pool": pool,
        "monthly_scale": monthly_scale,
        "r2_deterministic": float(1.0 - np.var(u, ddof=1) / var_y),
        "r2_total": float(1.0 - sigma ** 2 / var_y),
        "z_promo": detrended_promo_driver(df_base["PromotionAmount"].to_numpy(dtype=float)),
        "skewness": float(stats.skew(y_all, bias=False)),
        "kurtosis": float(stats.kurtosis(y_all, fisher=True, bias=False)),
    }


def base_mean_path(n: int, base_stats: dict) -> np.ndarray:
    X = fourier_design(np.arange(n), base_stats["season_period"], base_stats["n_harmonics"],
                       trend_cap=base_stats["n"] - 1)
    return X @ base_stats["beta"]


# ==============================================================================
# 2. LÕI MÔ PHỎNG
# ==============================================================================

def _simulate_core(m_t: np.ndarray, scale_t: np.ndarray, eps: np.ndarray, eps_burn: np.ndarray,
                   phi_post: np.ndarray, phi_pre: np.ndarray, tau: int | None, burn_scale: float):
    """Chạy AR với hệ số phi_pre trước tau, phi_post từ tau (burn-in thuộc chế độ đầu). Trả về (y, cond_mean, n_clipped)."""
    n, nb = len(m_t), len(eps_burn)
    switch = 0 if tau is None else nb + tau
    shocks = np.concatenate([eps_burn * burn_scale, eps * scale_t])
    u = np.zeros(nb + n)
    cond = np.zeros(nb + n)
    for i in range(nb + n):
        phi = phi_pre if i < switch else phi_post
        if len(phi):
            lags = u[max(0, i - len(phi)):i][::-1]
            cond[i] = float(phi[:len(lags)] @ lags)
        u[i] = cond[i] + shocks[i]
    y_raw = m_t + u[nb:]
    n_clipped = int(np.sum(y_raw < 0.0))
    if n_clipped:
        warnings.warn(f"{n_clipped} giá trị âm bị clip về 0.", RuntimeWarning)
    return np.maximum(y_raw, 0.0), m_t + cond[nb:], n_clipped


def _driver_series(df_template: pd.DataFrame, n: int, base_stats: dict, lag: int) -> np.ndarray:
    if n <= base_stats["n"]:
        z = base_stats["z_promo"][:n]
    else:
        assert len(df_template) >= n, "df_template phải có ít nhất n dòng (dùng bảng ngoại sinh đã mở rộng)."
        z = detrended_promo_driver(df_template["PromotionAmount"].to_numpy(dtype=float)[:n])
    g = np.tanh(z)
    g = (g - g.mean()) / (g.std() + 1e-12)
    out = np.zeros(n)
    out[lag:] = g[:n - lag] if lag > 0 else g
    return out


def simulate_by_key(key: str, df_template: pd.DataFrame, n: int, base_stats: dict, seed: int = 42,
                    meta: dict | None = None, **overrides):
    """
    Sinh một realization của kịch bản `key`. Trả về (y, extra):
      extra = None | {"break_t", "shift"} (s4) | {"break_t", "slope_change"} (s5) | {"exog_r2"} (s6) | {...}
    Nếu `meta` là dict, hàm điền thông tin cần cho OracleDGP.
    """
    if key not in SCENARIO_KEYS:
        raise ValueError(f"Kịch bản không hợp lệ: {key}")
    prm = {**DEFAULT_SCENARIO_PARAMS, **overrides}
    rng_noise = np.random.default_rng([seed, 0])
    rng_scn = np.random.default_rng([seed, 1])

    k = float(prm["noise_scale"])
    sigma_t = sigma_path(n, base_stats) * k
    m_t = base_mean_path(n, base_stats).copy()
    s_t = np.ones(n)
    pool = base_stats["innov_pool"]
    extra = None

    if key == "s3":
        nu = float(prm["t_df"])
        c = np.sqrt((nu - 2.0) / nu)
        eps_burn = rng_noise.standard_t(nu, size=BURN_IN) * c
        eps = rng_noise.standard_t(nu, size=n) * c
        innov = ("t", nu)
    else:
        eps_burn = rng_noise.choice(pool, size=BURN_IN, replace=True)
        eps = rng_noise.choice(pool, size=n, replace=True)
        innov = ("empirical", pool)

    if key == "s2":
        h = base_stats["monthly_scale"] ** float(prm["hetero_power"])
        h = h / np.sqrt(np.mean(h ** 2))
        s_t = h[month_index(n, base_stats["start_month"]) - 1]
        extra = {"max_min_ratio": float(h.max() / h.min())}

    elif key == "s4":
        lo, hi = prm["shift_window"]
        tau = int(rng_scn.integers(int(lo * n), max(int(hi * n), int(lo * n) + 1)))
        delta = float(rng_scn.choice([-1.0, 1.0]) * rng_scn.uniform(*prm["shift_frac"]) * base_stats["mean"])
        m_t[tau:] += delta
        extra = {"break_t": tau, "shift": delta}

    elif key == "s5":
        lo, hi = prm["break_window"]
        tau = int(rng_scn.integers(int(lo * n), max(int(hi * n), int(lo * n) + 1)))
        d_slope = float(rng_scn.choice([-1.0, 1.0]) * rng_scn.uniform(*prm["slope_frac"]) * base_stats["mean"])
        m_t += d_slope * np.maximum(0, np.arange(n) - tau)
        extra = {"break_t": tau, "slope_change": d_slope}

    elif key == "s6":
        beta = float(prm["exog_beta"])
        drv = _driver_series(df_template, n, base_stats, int(prm["exog_lag"]))
        m_t += beta * sigma_t * drv
        extra = {"exog_beta": beta, "exog_share_of_innov_var": beta ** 2 / (1.0 + beta ** 2)}

    scale_t = sigma_t * s_t
    tau = base_stats["regime_break"]
    burn_scale = float(np.mean(scale_t[:tau or n]))
    y, cond_mean, n_clipped = _simulate_core(m_t, scale_t, eps, eps_burn, base_stats["phi"], base_stats["phi_pre"],
                                             tau, burn_scale)

    if meta is not None:
        meta.update(key=key, n=n, cond_mean=cond_mean, scale=scale_t, innov=innov,
                    n_clipped=n_clipped, params=prm, extra=extra)
    return y, extra


# ==============================================================================
# 3. ORACLE: DỰ BÁO MỘT BƯỚC TỐI ƯU THEO ĐÚNG DGP
# ==============================================================================

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

def scenario_specifications(base_stats: dict, **overrides) -> dict:
    prm = {**DEFAULT_SCENARIO_PARAMS, **overrides}
    bs = base_stats
    k = prm["noise_scale"]
    h = bs["monthly_scale"] ** prm["hetero_power"]
    h = h / np.sqrt(np.mean(h ** 2))
    base_params = {
        "season_period_P": bs["season_period"],
        "n_harmonics_K (BIC)": bs["n_harmonics"],
        "trend_slope": f"{bs['trend_slope']:,.1f} / tháng",
        "ar_order_p (BIC)": bs["ar_order"],
        "phi": [round(float(x), 3) for x in bs["phi"]],
        "ar_spectral_radius": round(bs["ar_spectral_radius"], 3),
        "regime_break_tau": bs["regime_break"] if bs["regime_break"] is not None else "không (một chế độ)",
        "phi_pre (trước tau)": [round(float(x), 3) for x in bs["phi_pre"]],
        "sigma_innov_pre": f"{bs['sigma_pre']:,.0f} (x noise_scale={k})",
        "sigma_innov_post": f"{bs['sigma_post']:,.0f} (x noise_scale={k})",
        "R2_deterministic / R2_total (calib)": f"{bs['r2_deterministic']:.2f} / {bs['r2_total']:.2f}",
        "calibration_sample": f"y[0:{bs['calib_n']}]",
    }
    base_formula = (r"y_t=\max(0,m_t+u_t),\ m_t=a+bt+\sum_{k\le K}[c_k\sin\tfrac{2\pi kt}{P}+d_k\cos\tfrac{2\pi kt}{P}],"
                    r"\ u_t=\sum_{j\le p}\phi_j u_{t-j}+k\sigma_t s_t\varepsilon_t,"
                    r"\ \sigma_t=\sigma_{\text{pre}}\mathbb 1[t<\tau_\sigma]+\sigma_{\text{post}}\mathbb 1[t\ge\tau_\sigma]")
    return {
        "s1": {"name": SCENARIO_NAMES["s1"], "dgp_formula": base_formula + r",\ \varepsilon_t\sim\hat F_{\text{resid}}",
               "parameters": base_params,
               "rationale": "Kịch bản nền tái tạo cấu trúc trend/mùa vụ/tự tương quan của dữ liệu thật; "
                            "đo năng lực khai thác cấu trúc động khi nhiễu có hình dạng thực tế."},
        "s2": {"name": SCENARIO_NAMES["s2"], "dgp_formula": r"\text{Base},\ s_t=h_{\text{month}(t)}^{\gamma}",
               "parameters": {"gamma (hetero_power)": prm["hetero_power"],
                              "max/min scale ratio": round(float(h.max() / h.min()), 2)},
               "rationale": "Kiểm tra khả năng co giãn độ rộng khoảng phân vị theo tháng."},
        "s3": {"name": SCENARIO_NAMES["s3"], "dgp_formula": r"\text{Base},\ \varepsilon_t\sim t_\nu\sqrt{(\nu-2)/\nu}",
               "parameters": {"nu (t_df)": prm["t_df"]},
               "rationale": "Kiểm tra độ vững của phân vị đuôi (P5/P95) và chi phí tồn kho khi nhiễu đuôi dày."},
        "s4": {"name": SCENARIO_NAMES["s4"], "dgp_formula": r"\text{Base},\ m_t\mathrel{+}=\delta\,\mathbb 1[t\ge\tau]",
               "parameters": {"|delta| / mean": prm["shift_frac"], "tau / n": prm["shift_window"],
                              "sign": "ngẫu nhiên ±"},
               "rationale": "Concept drift xảy ra trong giai đoạn test; đo khả năng thích ứng (vd. RevIN) khi mức nhu cầu đổi đột ngột."},
        "s5": {"name": SCENARIO_NAMES["s5"], "dgp_formula": r"\text{Base},\ m_t\mathrel{+}=\Delta_b\max(0,t-\tau)",
               "parameters": {"|Delta_b| / mean per month": prm["slope_frac"], "tau / n": prm["break_window"],
                              "sign": "ngẫu nhiên ±"},
               "rationale": "Gãy trend quanh ranh giới train/val: mô hình phải ngoại suy độ dốc mới."},
        "s6": {"name": SCENARIO_NAMES["s6"],
               "dgp_formula": r"\text{Base},\ m_t\mathrel{+}=\beta k\sigma_t\,\tilde g(z_{t-\ell}),\ \tilde g=\text{std}(\tanh z)",
               "parameters": {"beta (exog_beta)": prm["exog_beta"], "lag": prm["exog_lag"],
                              "driver": "PromotionAmount khử trend tuyến tính, chuẩn hóa median/IQR",
                              "share of innovation variance": round(prm["exog_beta"] ** 2 / (1 + prm["exog_beta"] ** 2), 2)},
               "rationale": "Đo lợi ích của mixing đa biến: chỉ mô hình nhìn thấy biến ngoại sinh trễ mới khai thác được phần tín hiệu này."},
    }


# ==============================================================================
# 5. KIỂM CHỨNG: ĐỘ TRUNG THỰC SO VỚI DỮ LIỆU THẬT + ĐỘ CHUẨN CỦA ORACLE
# ==============================================================================

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

def extend_exogenous_features(df_base: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Mở rộng chuỗi ngoại sinh khi n_samples > len(df_base) (lặp lại + nhiễu 1% std)."""
    rng = np.random.default_rng(seed)
    n_base = len(df_base)
    exog_cols = [c for c in df_base.columns if c not in ("Month", "Quantity")]
    start_date = pd.to_datetime(df_base["Month"].iloc[0]) if "Month" in df_base.columns else pd.to_datetime("2015-01-01")
    months = pd.date_range(start=start_date, periods=n_samples, freq="MS").strftime("%Y-%m")
    df_ext = pd.DataFrame({"Month": months})
    repeats = int(np.ceil(n_samples / n_base))
    for col in exog_cols:
        base_vals = df_base[col].to_numpy()
        df_ext[col] = np.tile(base_vals, repeats)[:n_samples] + rng.normal(0, 0.01 * (np.std(base_vals) + 1e-8), size=n_samples)
    return df_ext


def plot_simulation_comparison(df_base: pd.DataFrame, sim_dict: dict[str, np.ndarray], save_dir: Path):
    """Time series, KDE và ACF của 6 kịch bản so với dữ liệu thật."""
    save_dir.mkdir(parents=True, exist_ok=True)
    real_q = df_base["Quantity"].to_numpy(dtype=float)
    n = len(real_q)
    t = np.arange(n)
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    for ax, key, col in zip(axes.flatten(), SCENARIO_KEYS, colors):
        ax.plot(t, real_q, label="Real", color="gray", alpha=0.5, linestyle="--", linewidth=1.5)
        ax.plot(t, sim_dict[key][:n], label=SCENARIO_NAMES[key], color=col, linewidth=1.6)
        ax.axvspan(int(0.8 * n), n, color="gray", alpha=0.08)
        ax.set_title(SCENARIO_NAMES[key], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    ts_path = save_dir / "simulated_scenarios_time_series.png"
    plt.savefig(ts_path, dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    lo = min(real_q.min(), min(v.min() for v in sim_dict.values()))
    hi = max(real_q.max(), max(v.max() for v in sim_dict.values()))
    x_grid = np.linspace(lo, hi, 400)
    kde_real = stats.gaussian_kde(real_q)(x_grid)
    for ax, key, col in zip(axes.flatten(), SCENARIO_KEYS, colors):
        sim_q = sim_dict[key][:n]
        kde_sim = stats.gaussian_kde(sim_q)(x_grid)
        ax.plot(x_grid, kde_real, label="Real KDE", color="gray", linestyle="--", linewidth=2)
        ax.fill_between(x_grid, 0, kde_sim, color=col, alpha=0.35, label="Sim KDE")
        ax.text(0.03, 0.80, f"Mean: {sim_q.mean():,.0f}\nStd : {sim_q.std(ddof=1):,.0f}\n"
                            f"Skew: {stats.skew(sim_q):.2f}\nKurt: {stats.kurtosis(sim_q):.2f}",
                transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.set_title(SCENARIO_NAMES[key], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    dist_path = save_dir / "simulated_scenarios_distributions.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()

    lags = np.arange(1, 37)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, sharey=True)
    acf_real = [_acf(real_q, L) for L in lags]
    for ax, key, col in zip(axes.flatten(), SCENARIO_KEYS, colors):
        ax.bar(lags - 0.2, acf_real, width=0.4, color="gray", alpha=0.6, label="Real")
        ax.bar(lags + 0.2, [_acf(sim_dict[key][:n], L) for L in lags], width=0.4, color=col, label="Sim")
        ax.axhline(1.96 / np.sqrt(n), color="k", lw=0.6, ls=":")
        ax.axhline(-1.96 / np.sqrt(n), color="k", lw=0.6, ls=":")
        ax.set_title(SCENARIO_NAMES[key], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    acf_path = save_dir / "simulated_scenarios_acf.png"
    plt.savefig(acf_path, dpi=200)
    plt.close()

    print(f"[OK] Time series : {ts_path}")
    print(f"[OK] Distribution: {dist_path}")
    print(f"[OK] ACF         : {acf_path}")


def generate_and_save_simulated_datasets(df_base: pd.DataFrame, n_samples: int | None = None,
                                         seed: int = 42, plot: bool = True, save_oracle: bool = False,
                                         base_stats: dict | None = None, **overrides) -> dict[str, Path]:
    """Sinh và lưu 6 file data_simulated_s{k}.csv (giữ Month & cột ngoại sinh, chỉ thay Quantity)."""
    n_base = len(df_base)
    n_eff = n_samples if (n_samples is not None and n_samples > 0) else n_base
    bs = base_stats or get_base_statistics(df_base)
    df_template = df_base.copy() if n_eff == n_base else extend_exogenous_features(df_base, n_eff, seed)

    print("=" * 95)
    print(f"SINH 6 KỊCH BẢN (N={n_eff}, seed={seed}) | P={bs['season_period']} K={bs['n_harmonics']} "
          f"AR p={bs['ar_order']} phi={np.round(bs['phi'], 3).tolist()} tau={bs['regime_break']} "
          f"sigma_pre={bs['sigma_pre']:,.0f} sigma_post={bs['sigma_post']:,.0f}")
    print("=" * 95)

    sims, saved = {}, {}
    for key in SCENARIO_KEYS:
        meta = {}
        y, extra = simulate_by_key(key, df_template, n_eff, bs, seed, meta=meta, **overrides)
        sims[key] = y
        df_sim = df_template.copy()
        if "Quantity" in df_sim.columns:
            df_sim["Quantity"] = y
        else:
            df_sim.insert(1, "Quantity", y)
        path = RAW_DATA_DIR / f"data_simulated_{key}.csv"
        df_sim.to_csv(path, index=False)
        saved[key] = path
        if n_eff != n_base:
            df_sim.to_csv(RAW_DATA_DIR / f"data_simulated_{key}_{n_eff}.csv", index=False)
        print(f"[{key.upper()}] {SCENARIO_NAMES[key]} -> {path.name}")
        print(f"      Mean={y.mean():,.0f} Std={y.std(ddof=1):,.0f} Min={y.min():,.0f} Max={y.max():,.0f} "
              f"ACF24={_acf(y, 24):.2f} extra={extra}")
        if save_oracle:
            orc = OracleDGP.forecast_path(meta)
            oracle_dir = RESULTS_DIR / "oracle"
            oracle_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(orc, index=df_template["Month"]).to_csv(oracle_dir / f"oracle_{key}.csv")

    if plot:
        plot_simulation_comparison(df_base, sims, RESULTS_DIR)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh dữ liệu mô phỏng hiệu chỉnh theo động học chuỗi thật.")
    parser.add_argument("--n_samples", type=int, default=None, help="Số mẫu (mặc định = số dòng dữ liệu gốc).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--monte_carlo", action="store_true", help="Kiểm chứng độ trung thực + Oracle trên nhiều seed.")
    parser.add_argument("--n_mc_seeds", type=int, default=100)
    parser.add_argument("--save_oracle", action="store_true")
    parser.add_argument("--season_period", type=int, default=24)
    parser.add_argument("--max_ar_order", type=int, default=6)
    parser.add_argument("--calib_end", type=int, default=None, help="Chỉ hiệu chỉnh trên y[:calib_end].")
    parser.add_argument("--regime_break", type=parse_regime_break, default="auto",
                        help="Điểm đổi chế độ nhiễu: auto, số nguyên t, hoặc none (một chế độ).")
    parser.add_argument("--noise_scale", type=float, default=DEFAULT_SCENARIO_PARAMS["noise_scale"])
    parser.add_argument("--exog_beta", type=float, default=DEFAULT_SCENARIO_PARAMS["exog_beta"])
    args = parser.parse_args()

    df_raw = pd.read_csv(BASE_DATA_PATH)
    bs = get_base_statistics(df_raw, season_period=args.season_period, max_ar_order=args.max_ar_order,
                             calib_end=args.calib_end, regime_break=args.regime_break)
    ov ={"noise_scale": args.noise_scale, "exog_beta": args.exog_beta}
    if args.monte_carlo:
        run_monte_carlo_validation(df_raw, n_seeds=args.n_mc_seeds, base_stats=bs, **ov)
    else:
        generate_and_save_simulated_datasets(df_raw, n_samples=args.n_samples, seed=args.seed,
                                             plot=not args.no_plot, save_oracle=args.save_oracle,
                                             base_stats=bs, **ov)
