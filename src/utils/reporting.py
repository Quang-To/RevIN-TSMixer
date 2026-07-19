# src/utils/reporting.py

import numpy as np

def reconstruct(flat_arr, seq_length: int) -> np.ndarray:
    """Reconstruct a continuous time series from overlapping sliding windows."""
    flat_arr = np.asarray(flat_arr).flatten()
    n_samples = len(flat_arr) // seq_length
    if n_samples == 0:
        return np.array([])
    reshaped = flat_arr.reshape(n_samples, seq_length)
    return np.concatenate([reshaped[0], reshaped[1:, -1]])


def print_run_summary(result: dict, scenario: int, seq_length: int) -> None:
    """Print simplified scenario results, inventory cost details, and decomposition diagnostics."""
    m = result["metrics"]
    tc_components = m.get('TC_components', {})

    print("\n" + "=" * 49)
    print(f"SCENARIO {scenario}")
    print("=" * 49)
    print(f"MAE      : {m['MAE']:.2f}")
    print(f"RMSE     : {m['RMSE']:.2f}")
    print(f"MAPE     : {m['MAPE']:.4f} %")
    print()
    print("Inventory")
    print(f"TC_min   : {m.get('TC_min', 0):.2f}")
    print(f"Ordering : {tc_components.get('ordering_cost_total', 0):.2f}")
    print(f"Holding  : {tc_components.get('holding_cost_total', 0):.2f}")
    print(f"Shortage : {tc_components.get('shortage_cost_total', 0):.2f}")
    print(f"ROP      : {tc_components.get('reorder_point', 0):.2f}")
    print("=" * 49)

    decomp = result.get('decomp_components')
    if decomp:
        target = reconstruct(decomp.get("raw_target", []), seq_length)
        trend = reconstruct(decomp.get("raw_trend", []), seq_length)
        seasonal = reconstruct(decomp.get("raw_seasonal", []), seq_length)
        residual = reconstruct(decomp.get("raw_residual", []), seq_length)

        if len(target) > 0 and len(trend) > 0 and len(seasonal) > 0 and len(residual) > 0:
            print("\nDecomposition info:")
            print(f"Input target mean: {target.mean():.2f}, std: {target.std():.2f}")
            print(f"trend mean: {trend.mean():.2f}, std: {trend.std():.2f}")
            print(f"seasonal mean: {seasonal.mean():.2f}, std: {seasonal.std():.2f}")
            print(f"residual mean: {residual.mean():.2f}, std: {residual.std():.2f}")
