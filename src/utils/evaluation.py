"""
src/utils/evaluation.py

Post-training helpers: unpack train_walk_forward results and print summary.
"""



def unpack_results(run_result: dict) -> dict:
    """
    Extract final test results from train_walk_forward's return value.

    train_walk_forward returns:
        { "val_folds": [...], "test": { "metrics", "test_pred", "test_true", "test_indices", "best_val" } }

    Returns a flat dict with keys:
        preds, trues, indices, metrics, tc_min, c_s_star
    """
    test = run_result["test"]
    return {
        "preds":    test["test_pred"],
        "trues":    test["test_true"],
        "indices":  test["test_indices"],
        "metrics":  test["metrics"],
        "tc_min":   test["metrics"]["TC_min"],
        "c_s_star": test["metrics"]["c_s_star"],
    }


import numpy as np


def print_summary(result: dict, scenario: int, seed: int) -> None:
    """Print overall metrics and a preview table of predictions."""
    m = result["metrics"]

    def _preview(name: str, values, n: int = 8) -> None:
        arr = np.asarray(values).flatten()
        if arr.size == 0:
            return
        preview = ", ".join(f"{v:.2f}" for v in arr[:n])
        suffix = " ..." if arr.size > n else ""
        print(f"  {name:<16}: [{preview}{suffix}]")

    print("\n" + "=" * 60)
    print(f"SCENARIO {scenario} — COMPLETE  |  SEED: {seed}")
    print("=" * 60)
    print(f"MAE      : {m['MAE']:.2f}")
    print(f"RMSE     : {m['RMSE']:.2f}")
    print(f"MAPE (%) : {m['MAPE']:.4f}")
    # Lấy TC_min từ nhiều nguồn, ưu tiên key thường gặp
    tc = result.get('tc_min')
    if tc is None:
        tc = result.get('TC_min')
    if tc is None and 'metrics' in result:
        tc = result['metrics'].get('TC_min')
    if tc is None:
        tc = 0
    print(f"TC_min   : {tc:.2f}")
    cs = result.get('c_s_star')
    if cs is None and 'metrics' in result:
        cs = result['metrics'].get('c_s_star', 0)
    if cs is None:
        cs = 0
    print(f"c_s*     : {cs:.4f}")
    # If available, print TC components
    tc_components = None
    if 'TC_components' in m:
        tc_components = m.get('TC_components')
    elif 'TC_components' in result:
        tc_components = result.get('TC_components')

    if tc_components:
        print("\nTC components:")
        print(f"  Ordering cost : {tc_components.get('ordering_cost_total', 0):.2f}")
        print(f"  Holding cost  : {tc_components.get('holding_cost_total', 0):.2f}")
        print(f"  Shortage cost : {tc_components.get('shortage_cost_total', 0):.2f}")
        print(f"  q*            : {tc_components.get('q_star', 0):.2f}")
        print(f"  Safety stock  : {tc_components.get('safety_stock', 0):.2f}")
        print(f"  Reorder point : {tc_components.get('reorder_point', 0):.2f}")

    # Forecast summary
    preds_arr = np.array(result.get('preds', []))
    if preds_arr.size:
        print("\nForecast summary:")
        print(f"  Forecast mean : {preds_arr.mean():.2f}")
        print(f"  Forecast std  : {preds_arr.std():.2f}")

    # Decomposition diagnostics (if available)
    decomp = result.get('decomp_components') or m.get('decomp_components')
    if decomp:
        raw_target = np.asarray(decomp.get('raw_target', [])).flatten()
        raw_trend = np.asarray(decomp.get('raw_trend', [])).flatten()
        raw_seasonal = np.asarray(decomp.get('raw_seasonal', [])).flatten()
        raw_residual = np.asarray(decomp.get('raw_residual', [])).flatten()

        if min(len(raw_target), len(raw_trend), len(raw_seasonal), len(raw_residual)) > 0:
            n_raw = min(len(raw_target), len(raw_trend), len(raw_seasonal), len(raw_residual))
            raw_target = raw_target[:n_raw]
            raw_trend = raw_trend[:n_raw]
            raw_seasonal = raw_seasonal[:n_raw]
            raw_residual = raw_residual[:n_raw]

            print("\nRaw decomposition diagnostics:")
            print(f"  Input target mean   : {raw_target.mean():.2f}, std: {raw_target.std():.2f}")
            print(f"  Raw trend mean      : {raw_trend.mean():.2f}, std: {raw_trend.std():.2f}")
            print(f"  Raw seasonal mean   : {raw_seasonal.mean():.2f}, std: {raw_seasonal.std():.2f}")
            print(f"  Raw residual mean   : {raw_residual.mean():.2f}, std: {raw_residual.std():.2f}")
            print("  Raw value previews:")
            _preview("target", raw_target)
            _preview("trend", raw_trend)
            _preview("seasonal", raw_seasonal)
            _preview("residual", raw_residual)

            raw_recon = raw_trend + raw_seasonal + raw_residual
            raw_diff = raw_target - raw_recon
            print(f"  Raw recon mean abs diff : {np.mean(np.abs(raw_diff)):.6f}")

            trend_scale = np.mean(np.abs(raw_trend))
            seasonal_scale = np.mean(np.abs(raw_seasonal))
            target_scale = np.mean(np.abs(raw_target))
            print(f"  |trend| mean / |target| mean   : {trend_scale / (target_scale + 1e-8):.4f}")
            print(f"  |seasonal| mean / |target| mean: {seasonal_scale / (target_scale + 1e-8):.4f}")
            if trend_scale < target_scale * 0.2:
                print("[Warning] Raw trend scale is much smaller than target scale; decomposition may be collapsing trend.")

        trend_branch = np.asarray(decomp.get('trend_branch', [])).flatten()
        seasonal_branch = np.asarray(decomp.get('seasonal_branch', [])).flatten()
        combined = np.asarray(decomp.get('combined', [])).flatten()
        n = min(len(trend_branch), len(seasonal_branch), len(combined))
        if n > 0:
            trend_branch = trend_branch[:n]
            seasonal_branch = seasonal_branch[:n]
            combined = combined[:n]
            print("\nBranch forecast diagnostics:")
            agg_method = decomp.get('aggregation_method')
            trend_w = decomp.get('trend_weight')
            seasonal_w = decomp.get('seasonal_weight')
            if agg_method is not None:
                print(f"  Aggregation method   : {agg_method}")
            if trend_w is not None or seasonal_w is not None:
                print(f"  Trend weight         : {trend_w}")
                print(f"  Seasonal weight      : {seasonal_w}")
            print(f"  Trend branch mean    : {trend_branch.mean():.2f}, std: {trend_branch.std():.2f}")
            print(f"  Seasonal branch mean : {seasonal_branch.mean():.2f}, std: {seasonal_branch.std():.2f}")
            print(f"  Combined mean        : {combined.mean():.2f}, std: {combined.std():.2f}")

            # Check reconstruction against the actual aggregation rule.
            if agg_method == "weighted" and trend_w is not None and seasonal_w is not None:
                weight_sum = (trend_w + seasonal_w) if (trend_w + seasonal_w) != 0 else 1.0
                w1 = trend_w / weight_sum
                w2 = seasonal_w / weight_sum
                recon = w1 * trend_branch + w2 * seasonal_branch
            elif agg_method == "sum" or agg_method is None:
                recon = trend_branch + seasonal_branch
            else:
                recon = trend_branch + seasonal_branch

            diff = combined - recon
            rel_err = np.mean(np.abs(diff)) / (np.mean(np.abs(recon)) + 1e-8)
            print(f"  Recon mean abs diff : {np.mean(np.abs(diff)):.4f} (rel {rel_err:.4f})")

            # Detect dropped trend: combined scale much smaller than components
            comp_scale = np.mean(np.abs(trend_branch)) + np.mean(np.abs(seasonal_branch))
            comb_scale = np.mean(np.abs(combined)) + 1e-8
            if comb_scale < comp_scale / 5.0:
                print("[Warning] Combined scale << component scales (possible missing trend in reconstruction).")
            if rel_err > 0.2:
                print("[Warning] Combined differs from trend+seasonal by >20% on average.")
    print("=" * 60)

    preds, trues, indices = result["preds"], result["trues"], result["indices"]
    min_len = min(len(preds), len(trues), len(indices))
    n_show = min(30, min_len)

    print("\n" + "=" * 80)
    print("TEST PREDICTIONS vs ACTUAL VALUES (unique, deduplicated)")
    print("=" * 80)
    print(f"{'Index':<8} {'Predicted':<15} {'Actual':<15} {'Diff':<15} {'% Error':<12}")
    print("-" * 80)
    for i in range(n_show):
        diff    = preds[i] - trues[i]
        pct_err = abs(diff) / (abs(trues[i]) + 1e-8) * 100
        print(f"{indices[i]:<8} {preds[i]:<15.2f} {trues[i]:<15.2f} {diff:<15.2f} {pct_err:<12.2f}%")
    if min_len > n_show:
        print(f"\n... ({min_len - n_show} more values)")
    if len(preds) != len(trues) or len(preds) != len(indices):
        print(f"[Warning] preds, trues, indices have mismatched lengths: preds={len(preds)}, trues={len(trues)}, indices={len(indices)}")
    print("=" * 80)