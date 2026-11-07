"""
evaluate_standard_dataset.py
============================
Đánh giá các bộ siêu tham số tối ưu (lấy từ simulation_hpo_best_params.csv)
trên tập dữ liệu chuẩn thực tế (data_TSI_v2.csv).

Cách dùng:
    # 1. Chạy tất cả mô hình và kịch bản trên tập chuẩn:
    python evaluate_standard_dataset.py

    # 2. Hoặc chỉ chạy kịch bản S2 và S5 cho 3 mô hình qua 3 seeds (0, 1, 2):
    python evaluate_standard_dataset.py --scenarios s2,s5 --models tsmixer,nbeats,nhits --seeds 0,1,2
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from run_simulation_benchmark import train_and_eval_single_seed, BASE_DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá siêu tham số từ HPO simulation trên tập dữ liệu chuẩn TSI_v2.")
    parser.add_argument("--params_file", type=str, default="results/simulation_hpo_best_params.csv",
                        help="Đường dẫn file HPO best params (mặc định: results/simulation_hpo_best_params.csv)")
    parser.add_argument("--scenarios", type=str, default="all",
                        help="Kịch bản HPO cần lấy tham số (e.g. all hoặc s1,s2,s5)")
    parser.add_argument("--models", type=str, default="all",
                        help="Mô hình cần test (e.g. all hoặc tsmixer,nbeats,nhits)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Danh sách seed đánh giá (mặc định: 0,1,2)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Số epochs tối đa")
    parser.add_argument("--patience", type=int, default=25,
                        help="Patience cho Early Stopping")
    parser.add_argument("--output_file", type=str, default="results/standard_dataset_benchmark_results.csv",
                        help="Đường dẫn lưu kết quả CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    params_path = Path(args.params_file)
    if not params_path.exists():
        print(f"❌ Không tìm thấy file siêu tham số: {params_path}")
        return

    df_params = pd.read_csv(params_path)
    df_base = pd.read_csv(BASE_DATA_PATH)
    y_real = df_base["Quantity"].to_numpy(dtype=float)

    eval_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Filter scenarios & models
    if args.scenarios.lower() != "all":
        req_sc = [s.strip().upper() for s in args.scenarios.split(",")]
        df_params = df_params[df_params["scenario"].str.upper().isin(req_sc)]

    if args.models.lower() != "all":
        req_md = [m.strip().upper() for m in args.models.split(",")]
        df_params = df_params[df_params["model"].str.upper().isin(req_md)]

    print("=" * 80)
    print(f"📊 ĐÁNH GIÁ SIÊU THAM SỐ HPO TRÊN TẬP CHUẨN THỰC TẾ (data_TSI_v2.csv)")
    print(f"• Dữ liệu: {BASE_DATA_PATH.name} (120 tháng | Train: 84, Val: 12, Test: 24)")
    print(f"• Tổng số cấu hình: {len(df_params)} cấu hình x {len(eval_seeds)} seeds")
    print("=" * 80)

    records = []

    for _, row in df_params.iterrows():
        sc = row["scenario"]
        md = row["model"]

        # Parse hyperparams
        params = {
            "lookback": int(row["lookback"]),
            "batch_size": int(row["batch_size"]),
            "learning_rate": float(row["learning_rate"]),
            "hidden_size": int(row["hidden_size"]),
            "n_layers": int(row["n_layers"]),
            "dropout": float(row["dropout"]),
            "revin_affine": bool(row["revin_affine"]),
        }
        if pd.notna(row.get("n_stacks")):
            params["n_stacks"] = int(row["n_stacks"])
        if pd.notna(row.get("n_blocks")):
            params["n_blocks"] = int(row["n_blocks"])

        seed_metrics = {"mae": [], "rmse": [], "mape": [], "pinball": [], "tc": [], "fill_rate": []}

        for seed in eval_seeds:
            res = train_and_eval_single_seed(
                model_name=md.lower(),
                params=params,
                y=y_real,
                df_full=df_base,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
            )
            for k in seed_metrics:
                seed_metrics[k].append(res[k if k != "tc" else "total_cost"])

        mean_mae = float(np.mean(seed_metrics["mae"]))
        mean_rmse = float(np.mean(seed_metrics["rmse"]))
        mean_mape = float(np.mean(seed_metrics["mape"]))
        mean_pinball = float(np.mean(seed_metrics["pinball"]))
        mean_tc = float(np.mean(seed_metrics["tc"]))
        mean_fill = float(np.mean(seed_metrics["fill_rate"]))

        print(f"[{sc}] {md:<8} | Lookback: {params['lookback']:2d} | MAE: {mean_mae:7.1f} | MAPE: {mean_mape:5.2f}% | Pinball: {mean_pinball:7.1f} | TC: {mean_tc:12,.0f} VND")

        records.append({
            "hpo_source_scenario": sc,
            "model": md,
            "lookback": params["lookback"],
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "hidden_size": params["hidden_size"],
            "n_layers": params["n_layers"],
            "dropout": params["dropout"],
            "revin_affine": params["revin_affine"],
            "mae_mean": mean_mae,
            "rmse_mean": mean_rmse,
            "mape_mean": mean_mape,
            "pinball_mean": mean_pinball,
            "total_cost_mean": mean_tc,
            "fill_rate_mean": mean_fill,
            "n_eval_seeds": len(eval_seeds),
        })

    df_out = pd.DataFrame(records)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu bảng kết quả chi tiết vào: {out_path}")


if __name__ == "__main__":
    main()
