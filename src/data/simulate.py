"""Public facade for calibrated demand simulation.

The implementation is split into calibration, scenario, oracle, and validation
modules. Existing imports from ``src.data.simulate`` remain supported.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Sequence
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "raw_data"
BASE_DATA_PATH = RAW_DATA_DIR / "data_TSI_v2.csv"
RESULTS_DIR = ROOT / "results"

from .calibration import (
    base_mean_path,
    detect_regime_break,
    detrended_promo_driver,
    fourier_design,
    get_base_statistics,
    month_index,
    parse_regime_break,
    sigma_path,
)
from .oracle import OracleDGP, innovation_quantile
from .scenarios import (
    DEFAULT_SCENARIO_PARAMS,
    SCENARIO_KEYS,
    SCENARIO_NAMES,
    scenario_specifications,
    simulate_by_key,
)
from .validation import _acf, run_monte_carlo_validation

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


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
