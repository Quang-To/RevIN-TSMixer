"""Scenario definitions and calibrated stochastic demand generation."""

import warnings

import numpy as np
import pandas as pd

from .calibration import base_mean_path, detrended_promo_driver, month_index, sigma_path

SCENARIO_KEYS = ("s1", "s2", "s3", "s4", "s5", "s6")

SCENARIO_NAMES = {
    "s1": "S1: Calibrated Base (Trend + Seasonal + AR, bootstrap noise)",
    "s2": "S2: Base + Seasonal Heteroscedasticity",
    "s3": "S3: Base + Heavy-tailed Innovations (Student-t)",
    "s4": "S4: Base + Level Shift in Test Period",
    "s5": "S5: Base + Trend Break",
    "s6": "S6: Base + Lagged Exogenous Driver (Promotion)",
}

DEFAULT_SCENARIO_PARAMS = {
    "noise_scale": 1.0,
    "hetero_power": 1.0,
    "t_df": 3.0,
    "shift_frac": (0.15, 0.25),
    "shift_window": (0.80, 0.90),
    "slope_frac": (0.004, 0.008),
    "break_window": (0.70, 0.80),
    "exog_beta": 1.0,
    "exog_lag": 1,
}

BURN_IN = 120

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
