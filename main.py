"""
main.py — Entry point for running the full training + evaluation pipeline.

This file only contains:
  1. Hyperparameter / scenario configuration
  2. Trainer instantiation
  3. Walk-forward training call
  4. Results aggregation + printing + visualisation
"""


import torch
from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.evaluation import print_summary
from src.utils.visualization import TrainingVisualizer

# ── Configuration ─────────────────────────────────────────────────────────────

SCENARIO   = 2
VAL_METRIC = "tc"          # "mape" for Scenario 1, "tc" for Scenario 2
SEED       = 42

# Model
SEQ_LENGTH = 12
PRED_LEN   = 3
N_BLOCK    = 3
FF_DIM     = 64
DROPOUT    = 0.1

# Training
BATCH_SIZE = 4
LR         = 1e-4
EPOCHS     = 2000
PATIENCE   = 100

# Inventory cost
HOLDING_COST  = 2.0
LEAD_TIME     = 2
ORDERING_COST = 50_000.0
FORECAST_HORIZON = 4

# ── Pipeline ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print(f"\n{'='*30} FINAL TEST {'='*30}")

    checkpoint = torch.load(f"checkpoints_optuna/s{SCENARIO}_rank1.pt")
    best_params = checkpoint["params"].copy()
    best_epoch = checkpoint.get("best_epoch", None)
    fold_best_epochs = checkpoint.get("fold_best_epochs", [])
    val_metric = checkpoint.get("val_metric", None)
    rank = checkpoint.get("rank", None)
    scenario_ckpt = checkpoint.get("scenario", None)
    for k in ["val_metric", "val_metric_type"]:
        if k in best_params:
            best_params.pop(k)

    print(f"Loaded checkpoint: rank={rank}, scenario={scenario_ckpt}, val_metric={val_metric}, best_epoch={best_epoch}, fold_best_epochs={fold_best_epochs}")

    TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer
    visualizer = TrainingVisualizer(save_dir="results")
    trainer = TrainerClass(
        **best_params,
        pred_len=PRED_LEN,
        epochs=EPOCHS,         
        patience=PATIENCE,     
        holding_cost=HOLDING_COST,
        ordering_cost=ORDERING_COST,
        seed=SEED,
        visualizer=visualizer,
    )

    walk_params = dict(
        seq_length=best_params["seq_length"],
        pred_length=PRED_LEN,
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=0.6,
        val_size=21,
        test_size=21,
        step=3,
    )
    run_result = trainer.train_and_test_with_best_hparams(
        walk_params,
        batch_size=best_params["batch_size"],
        best_epoch=best_epoch,  
        verbose=True
    )
    result = {
        "preds": run_result["test_pred"],
        "trues": run_result["test_true"],
        "indices": run_result.get("test_indices", []),
        "metrics": run_result["metrics"],
        "best_epoch": run_result.get("best_epoch", None),
        "fold_best_epochs": fold_best_epochs,
        "val_metric": val_metric,
        "rank": rank,
        "scenario": scenario_ckpt,
    }

    print_summary(result, scenario=SCENARIO, seed=SEED)
    visualizer.plot_training_history(scenario=SCENARIO)
    visualizer.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=SCENARIO)
    visualizer.plot_test_metrics(result["metrics"], scenario=SCENARIO)
    visualizer.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=SCENARIO)