# main.py — Entry point: load HPO checkpoint rồi train + evaluate

import logging
import warnings
import numpy as np

import config
from src.checkpoints import load_best_checkpoint_or_summary
from src.models.factory import build_trainer
from src.trainers import Scenario1Trainer, Scenario2Trainer
from src.utils.visualization import TrainingVisualizer
from src.utils.reporting import print_run_summary


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Bước 1: Load bộ siêu tham số tốt nhất từ Optuna HPO ───────────────────
    ckpt = load_best_checkpoint_or_summary(config.SCENARIO, config.MODEL, ranks=(1, 2, 3))
    best_params = ckpt.get("params", {}).copy()
    best_params.setdefault("use_log_return", config.USE_LOG_RETURN)
    for k in ("val_metric", "val_metric_type", "use_decomposition",
              "decomposition_method", "seasonal_period"):
        best_params.pop(k, None)

    logging.info(f"[Bước 1] Load Best HPO Params: rank={ckpt.get('rank')} | model={config.MODEL.upper()} | val_loss={ckpt.get('val_metric')}")
    logging.info(f"Params: { {k: best_params.get(k) for k in ['seq_length','batch_size','lr','dropout']} }")
    logging.info(f"Inventory: holding={config.HOLDING_COST}, shortage={config.SHORTAGE_COST}")
    logging.info(f"CR = {config.SHORTAGE_COST / (config.SHORTAGE_COST + config.HOLDING_COST):.4f}")

    TrainerClass = Scenario1Trainer if config.SCENARIO == 1 else Scenario2Trainer
    trainer = build_trainer(
        model_name   = config.MODEL,
        best_params  = best_params,
        TrainerClass = TrainerClass,
        visualizer   = TrainingVisualizer(save_dir="results"),
    )

    n = trainer.data_dict["n"]
    train_end = trainer.data_dict["train_end"]
    val_end = trainer.data_dict["val_end"]
    val_size = val_end - train_end
    test_size = n - val_end

    split_params = dict(
        seq_length       = best_params.get("seq_length",  config.SEQ_LENGTH),
        pred_length      = best_params.get("pred_length", config.PRED_LEN),
        forecast_horizon = config.FORECAST_HORIZON,
        train_size       = train_end,
        val_size         = val_size,
        test_size        = test_size,
    )
    logging.info(f"[Bước 2] Huấn luyện mô hình với Early Stopping trên tập Validation [{train_end}:{val_end}]...")
    run_result = trainer.train_with_early_stopping_and_test(
        split_params,
        batch_size = best_params.get("batch_size", config.BATCH_SIZE),
        verbose    = True,
        max_epochs = trainer.epochs,
        patience   = trainer.patience,
    )

    logging.info(f"[Bước 3] Tải checkpoint tốt nhất tại epoch {run_result['best_epoch']} và đánh giá trên tập Test [{val_end}:{n}].")
    metrics = run_result["metrics"]
    result  = {
        "preds":             run_result["test_pred"],
        "trues":             run_result["test_true"],
        "indices":           run_result.get("test_indices", []),
        "quantiles":         run_result.get("test_quantiles"),
        "metrics":           metrics,
        "best_epoch":        run_result.get("best_epoch"),
        "median_best_epoch": run_result.get("best_epoch"),
        "mean_best_epoch":   run_result.get("best_epoch"),
        "fold_best_epochs":  [run_result.get("best_epoch")],
        "val_metric":        ckpt.get("val_metric"),
        "rank":              ckpt.get("rank"),
        "scenario":          config.SCENARIO,
        "model":             config.MODEL,
    }

    # In kết quả tổng hợp
    print_run_summary(result, config.SCENARIO, best_params.get("seq_length", config.SEQ_LENGTH))



    # ── Bước 6: Vẽ biểu đồ ───────────────────────────────────────────────────
    viz = trainer.visualizer
    viz.plot_training_history(scenario=config.SCENARIO)
    viz.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=config.SCENARIO)
    viz.plot_test_metrics(metrics, scenario=config.SCENARIO)
    viz.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=config.SCENARIO)
    viz.plot_fan_chart(result["quantiles"], result["trues"], result["indices"], scenario=config.SCENARIO)
    viz.plot_calibration_curve(result["quantiles"], result["trues"], scenario=config.SCENARIO)
    viz.plot_expected_cost_curve(
        result["quantiles"], result["trues"],
        holding_cost=config.HOLDING_COST,
        shortage_cost=config.SHORTAGE_COST,
        scenario=config.SCENARIO,
    )
    viz.plot_eda()
    viz.update_model_comparison(config.MODEL, metrics, scenario=config.SCENARIO)


if __name__ == "__main__":
    main()
