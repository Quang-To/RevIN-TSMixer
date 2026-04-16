"""
main.py — Entry point for running the full training + evaluation pipeline.

This file only contains:
  1. Hyperparameter / scenario configuration
  2. Trainer instantiation
  3. Walk-forward training call
  4. Results aggregation + printing + visualisation
"""

from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.data.walk_forward import WalkForwardSplitter
from src.utils.evaluation import unpack_results, print_summary
from src.utils.visualization import TrainingVisualizer

# ── Configuration ─────────────────────────────────────────────────────────────

SCENARIO   = 2
VAL_METRIC = "tc"          # "mape" for Scenario 1, "tc" for Scenario 2
SEED       = 42

# Model
SEQ_LENGTH = 9
PRED_LEN   = 3
N_BLOCK    = 2
FF_DIM     = 128
DROPOUT    = 0.1

# Training
BATCH_SIZE = 2
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
    print(f"\n{'='*30} SEED: {SEED} {'='*30}")

    TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer

    visualizer = TrainingVisualizer(save_dir="results")
    trainer = TrainerClass(
      seq_length=SEQ_LENGTH,
      ff_dim=FF_DIM,
      dropout=DROPOUT,
      pred_len=PRED_LEN,
      n_block=N_BLOCK,
      batch_size=BATCH_SIZE,
      lr=LR,
      epochs=EPOCHS,
      patience=PATIENCE,
      holding_cost=HOLDING_COST,
      ordering_cost=ORDERING_COST,
      val_metric_type=VAL_METRIC,
      seed=SEED,
      visualizer=visualizer,
    )

    # Đúng chuẩn: train trên toàn bộ train+val, test trên test window cuối cùng
    walk_params = dict(
      seq_length=SEQ_LENGTH,
      pred_length=PRED_LEN,
      forecast_horizon=FORECAST_HORIZON,
      train_ratio=0.5,
      val_size=20,
      test_size=29,
      step=3,
    )
    run_result = trainer.train_and_test_with_best_hparams(walk_params, batch_size=BATCH_SIZE, verbose=True)
    result = {
      "preds": run_result["test_pred"],
      "trues": run_result["test_true"],
      "indices": run_result.get("test_indices", []),
      "metrics": run_result["metrics"],
    }

    print_summary(result, scenario=SCENARIO, seed=SEED)

    visualizer.plot_training_history(scenario=SCENARIO)
    visualizer.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=SCENARIO)
    visualizer.plot_test_metrics(result["metrics"], scenario=SCENARIO)
    visualizer.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=SCENARIO)