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



def print_summary(result: dict, scenario: int, seed: int) -> None:
    """Print overall metrics and a preview table of predictions."""
    m = result["metrics"]
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