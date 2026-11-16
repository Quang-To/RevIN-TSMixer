import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate as S

if "n_harmonics" not in inspect.signature(S.get_base_statistics).parameters:
    raise SystemExit(
        "simulate.get_base_statistics chưa có tham số n_harmonics. Sửa 2 chỗ trong simulate.py:\n"
        "  (1) thêm  n_harmonics: int | None = None  vào chữ ký hàm get_base_statistics\n"
        "  (2) thay  for K in range(0, season_period // 2 + 1):\n"
        "      bằng  for K in (range(0, season_period // 2 + 1) if n_harmonics is None else [int(n_harmonics)]):"
    )


# ------------------------------------------------------------------------------------------------
# Tiện ích
# ------------------------------------------------------------------------------------------------
def n_params(P: int, K: int) -> int:
    return S.fourier_design(np.arange(4), P, K).shape[1]


def wls_fit(X, y, tau, floor, n_iter=25):
    """Giống S._fit_mean nhưng có sàn cho sigma (tránh s_post -> 0 khi tham số ~ số điểm êm)."""
    n = len(y)
    if tau is None or tau <= 0 or tau >= n:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        s = max(float(np.sqrt(np.mean((y - X @ beta) ** 2))), floor)
        return beta, s, s, False
    pre = np.arange(n) < tau
    w = np.ones(n)
    hit = False
    for _ in range(n_iter):
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        r = y - X @ beta
        s_pre = float(np.sqrt(np.mean(r[pre] ** 2)))
        s_post = float(np.sqrt(np.mean(r[~pre] ** 2)))
        hit = hit or (s_pre < floor) or (s_post < floor)
        s_pre, s_post = max(s_pre, floor), max(s_post, floor)
        w = np.where(pre, 1.0 / s_pre, 1.0 / s_post)
    return beta, s_pre, s_post, hit


# ------------------------------------------------------------------------------------------------
# A. In-sample (đúng pipeline simulate.py)
# ------------------------------------------------------------------------------------------------
def insample_table(df, y, P, tau, Ks, max_ar):
    n = len(y)
    t = np.arange(n)
    n_post = n - (tau or 0)
    rows = []
    for K in Ks:
        X = S.fourier_design(t, P, K)
        _, _, _, neg2ll = S._fit_mean(X, y, tau)
        k = X.shape[1]
        bic = neg2ll + k * np.log(n)
        aicc = neg2ll + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
        bs = S.get_base_statistics(df, season_period=P, max_ar_order=max_ar, regime_break=tau, n_harmonics=K)
        rows.append(dict(K=K, n_params=k, params_per_post_pt=k / n_post, BIC=bic, AICc=aicc,
                         sigma_pre=bs["sigma_pre"], sigma_post=bs["sigma_post"], R2_det=bs["r2_deterministic"],
                         ar_p=bs["ar_order"], ar_rho_raw=bs["ar_spectral_radius_raw"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# B. Rolling-origin
# ------------------------------------------------------------------------------------------------
def rolling_cv(y, P, K, tau, n0, H, step, max_ar, floor_frac):
    n = len(y)
    t = np.arange(n)
    X_all = S.fourier_design(t, P, K)          # trend KHÔNG bị chặn: ngoại suy tuyến tính như một dự báo thật
    floor = floor_frac * float(np.std(y))
    out = []
    for o in range(n0, n - H + 1, step):
        te = np.arange(o, o + H)
        tau_f = tau if (tau is not None and tau < o) else None
        beta, s_pre, s_post, hit = wls_fit(X_all[:o], y[:o], tau_f, floor)
        m = X_all @ beta
        u = y - m
        e_det = y[te] - m[te]
        # AR(p) fit trên phần dư chuẩn hoá của chế độ cuối (giống pipeline), BIC chọn p
        a = tau_f if tau_f is not None else 0
        seg = u[a:o] / s_post
        max_p = int(min(max_ar, max(len(seg) // 6, 0)))
        p, phi = 0, np.zeros(0)
        if max_p >= 1 and len(seg) > max_p + 5:
            try:
                p, phi, _, _ = S._select_ar(seg, max_p)
            except Exception:
                p, phi = 0, np.zeros(0)
        pred = np.array([float(phi @ u[i - p:i][::-1]) if p > 0 else 0.0 for i in te])
        e_ar = y[te] - (m[te] + pred)
        out.append(dict(origin=o, floor_hit=hit, p=p, sigma_post_fold=s_post,
                        mae_det=np.mean(np.abs(e_det)), mse_det=np.mean(e_det ** 2),
                        mae_ar1=np.mean(np.abs(e_ar)), mse_ar1=np.mean(e_ar ** 2)))
    return pd.DataFrame(out)


def one_se_rule(loss_by_K: dict, n_eff: float):
    """Paired 1-SE: K nhỏ nhất mà mức chênh trung bình so với K tốt nhất <= 1 SE của chênh lệch."""
    Ks = sorted(loss_by_K)
    mean = {K: loss_by_K[K].mean() for K in Ks}
    best = min(Ks, key=lambda K: mean[K])
    chosen = best
    for K in Ks:
        d = loss_by_K[K] - loss_by_K[best]
        se = d.std(ddof=1) / np.sqrt(max(n_eff, 1.0)) if len(d) > 1 else np.inf
        if d.mean() <= se:
            chosen = K
            break
    return best, chosen, mean


# ------------------------------------------------------------------------------------------------
# D. Fidelity của S1
# ------------------------------------------------------------------------------------------------
def fidelity_S1(df, y_real, P, tau, Ks, max_ar, n_seeds):
    n = len(y_real)
    real = dict(std=y_real.std(ddof=1), a1=S._acf(y_real, 1), a12=S._acf(y_real, 12), a24=S._acf(y_real, 24))
    post = slice(tau or 0, n)
    rows = []
    for K in Ks:
        bs = S.get_base_statistics(df, season_period=P, max_ar_order=max_ar, regime_break=tau, n_harmonics=K)
        m = S.base_mean_path(n, bs)
        acc = {c: [] for c in ("std", "a1", "a12", "a24", "clip")}
        for s in range(n_seeds):
            meta = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _ = S.simulate_by_key("s1", df, n, bs, seed=s, meta=meta)
            acc["std"].append(y.std(ddof=1)); acc["a1"].append(S._acf(y, 1))
            acc["a12"].append(S._acf(y, 12)); acc["a24"].append(S._acf(y, 24))
            acc["clip"].append(meta["n_clipped"] / n)
        r = {c: float(np.mean(v)) for c, v in acc.items()}
        rows.append(dict(K=K, std_ratio=r["std"] / real["std"], acf1=r["a1"], acf12=r["a12"], acf24=r["a24"],
                         d_acf24=r["a24"] - real["a24"], clip_rate=r["clip"],
                         post_fit_rmse=float(np.sqrt(np.mean((y_real[post] - m[post]) ** 2)))))
    return pd.DataFrame(rows), real


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=S.BASE_DATA_PATH)
    ap.add_argument("--out", type=Path, default=S.RESULTS_DIR / "k_sensitivity")
    ap.add_argument("--period", type=int, default=24)
    ap.add_argument("--k_max", type=int, default=None, help="Mặc định P//2.")
    ap.add_argument("--regime_break", type=S.parse_regime_break, default="auto")
    ap.add_argument("--max_ar_order", type=int, default=6)
    ap.add_argument("--n0", type=int, default=60, help="Kích thước train nhỏ nhất của rolling-origin.")
    ap.add_argument("--H", type=int, default=12, help="Horizon mỗi origin.")
    ap.add_argument("--step", type=int, default=3, help="Bước dịch origin.")
    ap.add_argument("--floor_frac", type=float, default=0.01, help="Sàn sigma = floor_frac * std(y) trong CV.")
    ap.add_argument("--mc_seeds", type=int, default=50)
    ap.add_argument("--criterion", choices=["mae_ar1", "mae_det", "mse_ar1", "mse_det"], default="mae_ar1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    y = df["Quantity"].to_numpy(float)
    n, P = len(y), args.period
    tau = S.detect_regime_break(y, P) if args.regime_break == "auto" else args.regime_break
    Ks = list(range(0, (args.k_max if args.k_max is not None else P // 2) + 1))
    print(f"n={n}  P={P}  tau={tau} (đoạn 'êm' còn {n - (tau or 0)} điểm = {(n - (tau or 0)) / P:.1f} chu kỳ)")

    # ---- A
    ins = insample_table(df, y, P, tau, Ks, args.max_ar_order)
    K_bic = int(ins.loc[ins.BIC.idxmin(), "K"]); K_aicc = int(ins.loc[ins.AICc.idxmin(), "K"])

    # ---- B
    cv, per_origin = [], {}
    for K in Ks:
        r = rolling_cv(y, P, K, tau, args.n0, args.H, args.step, args.max_ar_order, args.floor_frac)
        per_origin[K] = r
        cv.append(dict(K=K, **{c: r[c].mean() for c in ("mae_det", "mae_ar1")},
                       rmse_det=np.sqrt(r.mse_det.mean()), rmse_ar1=np.sqrt(r.mse_ar1.mean()),
                       floor_hit_frac=r.floor_hit.mean()))
    cv = pd.DataFrame(cv)
    n_orig = len(per_origin[Ks[0]])
    n_eff = n_orig * args.step / args.H                      # số cửa sổ H không chồng lấn tương đương
    res = {}
    for crit in ("mae_det", "mae_ar1", "mse_det", "mse_ar1"):
        best, one, _ = one_se_rule({K: per_origin[K][crit].to_numpy() for K in Ks}, n_eff)
        res[crit] = (best, one)
    K_min, K_1se = res[args.criterion]

    # ---- C
    tab = ins.merge(cv, on="K")
    tab["optimism_ratio"] = tab["sigma_post"] / tab["rmse_ar1"]

    # ---- D
    Kshow = sorted(set([0, 2, 3, 4, K_bic, K_min, K_1se, Ks[-1]]) & set(Ks))
    fid, real = fidelity_S1(df, y, P, tau, Kshow, args.max_ar_order, args.mc_seeds)

    tab.to_csv(args.out / "k_selection_table.csv", index=False)
    fid.to_csv(args.out / "k_fidelity_S1.csv", index=False)

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:,.3f}")
    print("\n=== A+B+C: bảng theo K ===")
    print(tab[["K", "n_params", "params_per_post_pt", "BIC", "AICc", "sigma_post", "R2_det",
               "mae_det", "mae_ar1", "rmse_ar1", "optimism_ratio"]].to_string(index=False))
    print(f"\nRolling-origin: {n_orig} origins (n0={args.n0}, H={args.H}, step={args.step}) ~ {n_eff:.1f} cửa sổ độc lập")
    print("\n=== Lựa chọn K ===")
    print(f"  BIC (hiện tại)            : K={K_bic}")
    print(f"  AICc                      : K={K_aicc}")
    for crit, (b, o) in res.items():
        print(f"  rolling {crit:<8}  min: K={b:<2}  | 1-SE (paired): K={o}")
    print(f"\n=== D: fidelity S1 ({args.mc_seeds} seeds) | real: std={real['std']:,.0f} "
          f"ACF1={real['a1']:.2f} ACF12={real['a12']:.2f} ACF24={real['a24']:.2f} ===")
    print(fid.to_string(index=False))

    # ---- hình
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    a.plot(tab.K, tab.BIC - tab.BIC.min(), "o-", label="BIC - min")
    a.plot(tab.K, tab.AICc - tab.AICc.min(), "s--", label="AICc - min")
    a.set_title("In-sample criteria (WLS theo chế độ)"); a.set_xlabel("K"); a.legend(); a.grid(alpha=.3)

    a = ax[0, 1]
    for crit, mk in (("mae_ar1", "o-"), ("mae_det", "s--")):
        a.plot(tab.K, tab[crit], mk, label=crit)
    for v, c, lab in ((K_bic, "r", "BIC"), (K_min, "g", "CV-min"), (K_1se, "b", "CV-1SE")):
        a.axvline(v, color=c, ls=":", label=f"{lab}: K={v}")
    a.set_yscale("log"); a.set_title(f"Rolling-origin MAE (n0={args.n0}, H={args.H})")
    a.set_xlabel("K"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 0]
    a.plot(tab.K, tab.optimism_ratio, "o-", color="purple")
    a.axhline(1.0, color="k", lw=.8)
    a.set_title("Optimism: sigma_post in-sample / RMSE out-of-sample (<1: nhiễu bị đánh giá thấp)")
    a.set_xlabel("K"); a.grid(alpha=.3)

    a = ax[1, 1]
    t = np.arange(n); x0 = tau or 0
    a.plot(t[x0:], y[x0:], "k.-", alpha=.5, label="Real (sau tau)")
    for K in sorted(set([2, 4, K_1se, Ks[-1]]) & set(Ks)):
        bs = S.get_base_statistics(df, season_period=P, max_ar_order=args.max_ar_order, regime_break=tau, n_harmonics=K)
        a.plot(t[x0:], S.base_mean_path(n, bs)[x0:], label=f"m_t, K={K}")
    a.set_title("Đường trung bình tất định m_t theo K"); a.legend(fontsize=8); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(args.out / "k_selection.png", dpi=160); plt.close()
    print(f"\n[OK] {args.out}/k_selection_table.csv, k_fidelity_S1.csv, k_selection.png")


if __name__ == "__main__":
    main()