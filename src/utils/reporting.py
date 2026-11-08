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
    print(f"Pinball  : {m.get('Pinball_Loss', float('nan')):.2f}")
    for key in ("Pinball_P5", "Pinball_P10", "Pinball_P25", "Pinball_P50",
                "Pinball_P75", "Pinball_P90", "Pinball_P95"):
        if key in m:
            print(f"  {key:<12}: {m[key]:.2f}")
    print()
    print("Uncertainty & calibration")
    print(f"Coverage P5:", m.get('coverage_P5', float('nan')))
    print(f"Coverage P10:", m.get('coverage_P10', float('nan')))
    print(f"Coverage P25:", m.get('coverage_P25', float('nan')))
    print(f"Coverage P50:", m.get('coverage_P50', float('nan')))
    print(f"Coverage P75:", m.get('coverage_P75', float('nan')))
    print(f"Coverage P90:", m.get('coverage_P90', float('nan')))
    print(f"Coverage P95:", m.get('coverage_P95', float('nan')))
    print(f"ECE             : {m.get('ECE', float('nan')):.4f}")

    print(f"Coverage P5-P95 : {m.get('interval_coverage_P5_P95', float('nan')):.2%} (nominal 90%)")
    print(f"Coverage P10-P90: {m.get('interval_coverage_P10_P90', float('nan')):.2%} (nominal 80%)")
    print(f"Coverage P25-P75: {m.get('interval_coverage_P25_P75', float('nan')):.2%} (nominal 50%)")
    print()
    print("Inventory")
    print(f"Realized TC: {m.get('Realized_TC', m.get('TC_min', 0)):.2f}")
    print(f"Holding  :{tc_components.get('holding_cost_total', 0):.2f}")
    print(f"Shortage : {tc_components.get('shortage_cost_total', 0):.2f}")
    print(f"Overstock units : {tc_components.get('holding_units_total', 0):,.2f}")
    print(f"Shortage units  : {tc_components.get('shortage_units_total', 0):,.2f}")
    print(f"CR       : {tc_components.get('critical_ratio', float('nan')):.4f}")
    print(f"Actual mean: {tc_components.get('mu', float('nan')):.2f}")
    print(f"P50 mean : {tc_components.get('p50_forecast_mean', float('nan')):.2f}")
    print(f"S_t mean : {tc_components.get('s_level_mean', float('nan')):.2f}")
    print(f"Safety stock (z*sigma): {tc_components.get('safety_stock', float('nan')):.2f}")
    print(f"Fill rate: {m.get('fill_rate', float('nan')):.2%}")
    print(f"Stockout : {m.get('stockout_months', 0)} months")
    print("=" * 49)

    preds = np.asarray(result.get("preds", [])).reshape(-1)
    trues = np.asarray(result.get("trues", [])).reshape(-1)
    indices = np.asarray(result.get("indices", [])).reshape(-1)

    if preds.size:
        print("\nTest quantity forecasts (P50):")
        print(f"{'Step':>6} {'Index':>12} {'Predicted':>14} {'Actual':>14} {'Error':>14}")
        print("-" * 64)
        for step, predicted in enumerate(preds, start=1):
            index = indices[step - 1] if step <= indices.size else step
            if isinstance(index, (float, np.floating)) and index.is_integer():
                index = int(index)

            if step <= trues.size:
                actual = trues[step - 1]
                print(
                    f"{step:>6} {str(index):>12} {predicted:>14,.2f} "
                    f"{actual:>14,.2f} {predicted - actual:>14,.2f}"
                )
            else:
                print(f"{step:>6} {str(index):>12} {predicted:>14,.2f}")
        print("=" * 64)

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
