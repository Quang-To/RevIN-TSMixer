"""Module sinh du lieu mo phong (Simulate Data) cho 2 kich ban voi so luong mau tuy chon (mac dinh 1000 mau):
- Kich ban 1: Phan phoi chuan co dinh (Constant Sigma: Mean = 349,023.175, Std = 74,673.365)
- Kich ban 2: Phan phoi chuan voi sigma bien doi theo chu ky 12 thang:
    sigma_t = sigma_0 * (1 + alpha * sin(2 * pi * t / 12))
    alpha = (sigma_max - sigma_min) / (sigma_max + sigma_min) = 0.20803030132580913
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data" / "raw_data"
BASE_DATA_PATH = RAW_DATA_DIR / "data_TSI_v2.csv"

# Hang so mo phong theo thiet ke
MEAN = 349023.175
SIGMA_0 = 74673.36479332742
SIGMA_MIN = 61504.81780408455
SIGMA_MAX = 93816.3211411242
ALPHA = (SIGMA_MAX - SIGMA_MIN) / (SIGMA_MAX + SIGMA_MIN)  # 0.20803030132580913


def simulate_scenario_1(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Kich ban 1: Quantity ~ N(Mean, Sigma_0^2)."""
    np.random.seed(seed)
    return np.random.normal(loc=MEAN, scale=SIGMA_0, size=n_samples)


def simulate_scenario_2(n_samples: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Kich ban 2: Quantity_t ~ N(Mean, Sigma_t^2) voi Sigma_t = Sigma_0 * (1 + alpha * sin(2*pi*t / 12))."""
    np.random.seed(seed)
    t = np.arange(n_samples)
    sigma_t = SIGMA_0 * (1.0 + ALPHA * np.sin(2.0 * np.pi * t / 12.0))
    quantity = np.random.normal(loc=MEAN, scale=sigma_t)
    return quantity, sigma_t


def extend_exogenous_features(df_base: pd.DataFrame, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Keo dai cac dac trung ngoai sinh den n_samples bang cach lap chu ky co nhieu nhe."""
    np.random.seed(seed)
    n_base = len(df_base)
    exog_cols = [c for c in df_base.columns if c not in ("Month", "Quantity")]
    
    # Tao truc thoi gian Month cho du n_samples
    start_date = pd.to_datetime(df_base["Month"].iloc[0]) if "Month" in df_base.columns else pd.to_datetime("2015-01-01")
    months = pd.date_range(start=start_date, periods=n_samples, freq="MS").strftime("%Y-%m")
    
    df_ext = pd.DataFrame({"Month": months})
    
    # Mo rong cac cot ngoai sinh
    repeats = int(np.ceil(n_samples / n_base))
    for col in exog_cols:
        base_vals = df_base[col].to_numpy()
        tiled = np.tile(base_vals, repeats)[:n_samples]
        noise = np.random.normal(0, 0.01 * np.std(base_vals), size=n_samples)
        df_ext[col] = tiled + noise
        
    return df_ext


def generate_and_save_simulated_datasets(n_samples: int = 1000, base_path: Path = BASE_DATA_PATH, seed: int = 42) -> dict[str, Path]:
    """Tao 2 file CSV mo phong voi n_samples dong."""
    df_base = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()
    
    df_extended = extend_exogenous_features(df_base, n_samples=n_samples, seed=seed)

    # 1. Kich ban 1
    df_s1 = df_extended.copy()
    q_s1 = simulate_scenario_1(n_samples=n_samples, seed=seed)
    # Chen cot Quantity ngay sau Month
    df_s1.insert(1, "Quantity", q_s1)
    path_s1 = RAW_DATA_DIR / f"data_simulated_s1_{n_samples}.csv"
    path_s1_default = RAW_DATA_DIR / "data_simulated_s1.csv"
    df_s1.to_csv(path_s1, index=False)
    df_s1.to_csv(path_s1_default, index=False)

    # 2. Kich ban 2
    df_s2 = df_extended.copy()
    q_s2, sigma_t = simulate_scenario_2(n_samples=n_samples, seed=seed)
    df_s2.insert(1, "Quantity", q_s2)
    path_s2 = RAW_DATA_DIR / f"data_simulated_s2_{n_samples}.csv"
    path_s2_default = RAW_DATA_DIR / "data_simulated_s2.csv"
    df_s2.to_csv(path_s2, index=False)
    df_s2.to_csv(path_s2_default, index=False)

    print(f"[OK] Da tao tap du lieu Kich ban 1 ({n_samples} mau): {path_s1_default}")
    print(f"     Mean: {df_s1['Quantity'].mean():,.2f} | Std: {df_s1['Quantity'].std():,.2f}")
    print(f"[OK] Da tao tap du lieu Kich ban 2 ({n_samples} mau): {path_s2_default}")
    print(f"     Mean: {df_s2['Quantity'].mean():,.2f} | Min Sigma_t: {sigma_t.min():,.2f} | Max Sigma_t: {sigma_t.max():,.2f}")

    return {"s1": path_s1_default, "s2": path_s2_default}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated datasets.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of simulated samples (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    
    generate_and_save_simulated_datasets(n_samples=args.n_samples, seed=args.seed)
