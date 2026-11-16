"""Calibration utilities for the demand data-generating process."""

import numpy as np
import pandas as pd
from scipy import stats

AR_STABILITY_LIMIT = 0.97

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
                        calib_end: int | None = None, regime_break: str | int | None = "auto",
                        n_harmonics: int | None = None) -> dict:
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
    for K in (range(0, season_period // 2 + 1) if n_harmonics is None else [int(n_harmonics)]):
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
