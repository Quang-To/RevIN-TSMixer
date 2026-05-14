import torch
from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.evaluation import print_summary
from src.utils.visualization import TrainingVisualizer

# ── Configuration ─────────────────────────────────────────────────────────────

SCENARIO   = 2
MODEL      = "nhits"  # "tsmixer", "nbeats", or "nhits"
VAL_METRIC = "tc"          # "mape" for Scenario 1, "tc" for Scenario 2
SEED       = 42

# Model
SEQ_LENGTH = 12
PRED_LEN   = 3

# TSMixer specific
N_BLOCK    = 3
FF_DIM     = 64

# NBEATS specific
N_STACKS   = 3
N_LAYERS   = 3
LAYER_DIM  = 256

# NHITS specific
N_BLOCKS   = 1
HIDDEN_DIM = 128

# Common
DROPOUT    = 0.1

# Training
BATCH_SIZE = 2
LR         = 1e-4
EPOCHS     = 3000
PATIENCE   = 100

# Inventory cost
HOLDING_COST  = 2.0
LEAD_TIME     = 2
ORDERING_COST = 50_000.0
FORECAST_HORIZON = 4

# ── Pipeline ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print(f"\n{'='*30} FINAL TEST {'='*30}")

    checkpoint = torch.load(f"checkpoints_optuna/s{SCENARIO}_{MODEL}_rank1.pt")
    best_params = checkpoint["params"].copy()
    best_epoch = checkpoint.get("best_epoch", None)
    fold_best_epochs = checkpoint.get("fold_best_epochs", [])
    val_metric = checkpoint.get("val_metric", None)
    rank = checkpoint.get("rank", None)
    scenario_ckpt = checkpoint.get("scenario", None)
    model_ckpt = checkpoint.get("model_type", "tsmixer")
    for k in ["val_metric", "val_metric_type"]:
        if k in best_params:
            best_params.pop(k)

    print(f"Loaded checkpoint: rank={rank}, scenario={scenario_ckpt}, model={model_ckpt}, val_metric={val_metric}, best_epoch={best_epoch}, fold_best_epochs={fold_best_epochs}")

    TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer
    visualizer = TrainingVisualizer(save_dir="results")
    
    if MODEL == "tsmixer":
        trainer = TrainerClass(
            seq_length=best_params.get("seq_length", SEQ_LENGTH),
            ff_dim=best_params.get("ff_dim", FF_DIM),
            n_block=best_params.get("n_block", N_BLOCK),
            dropout=best_params.get("dropout", DROPOUT),
            pred_len=PRED_LEN,
            batch_size=best_params.get("batch_size", BATCH_SIZE),
            lr=best_params.get("lr", LR),
            epochs=EPOCHS,         
            patience=PATIENCE,     
            holding_cost=HOLDING_COST,
            ordering_cost=ORDERING_COST,
            lead_time=LEAD_TIME,
            seed=SEED,
            visualizer=visualizer,
            model_type="tsmixer",
        )
    elif MODEL == "nbeats":
        trainer = TrainerClass(
            seq_length=SEQ_LENGTH,
            n_stacks=N_STACKS,
            n_layers=N_LAYERS,
            layer_dim=LAYER_DIM,
            dropout=DROPOUT,
            pred_len=PRED_LEN,
            batch_size=BATCH_SIZE,
            lr=LR,
            epochs=EPOCHS,         
            patience=PATIENCE,     
            holding_cost=HOLDING_COST,
            ordering_cost=ORDERING_COST,
            lead_time=LEAD_TIME,
            seed=SEED,
            visualizer=visualizer,
            model_type="nbeats",
        )
    elif MODEL == "nhits":
        trainer = TrainerClass(
            seq_length=best_params.get("seq_length", SEQ_LENGTH),
            n_stacks=best_params.get("n_stacks", N_STACKS),
            n_blocks=best_params.get("n_blocks", N_BLOCKS),
            n_layers=best_params.get("n_layers", N_LAYERS),
            hidden_dim=best_params.get("hidden_dim", HIDDEN_DIM),
            dropout=best_params.get("dropout", DROPOUT),
            pred_len=PRED_LEN,
            batch_size=best_params.get("batch_size", BATCH_SIZE),
            lr=best_params.get("lr", LR),
            epochs=EPOCHS,         
            patience=PATIENCE,     
            holding_cost=HOLDING_COST,
            ordering_cost=ORDERING_COST,
            lead_time=LEAD_TIME,
            seed=SEED,
            visualizer=visualizer,
            model_type="nhits",
        )
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}. Must be 'tsmixer', 'nbeats', or 'nhits'")

    walk_params = dict(
        seq_length=SEQ_LENGTH,
        pred_length=PRED_LEN,
        forecast_horizon=FORECAST_HORIZON,
        train_ratio=0.6,
        val_size=21,
        test_size=21,
        step=3,
    )
    run_result = trainer.train_and_test_with_best_hparams(
        walk_params,
        batch_size=BATCH_SIZE,
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
        "model": model_ckpt,
    }

    print_summary(result, scenario=SCENARIO, seed=SEED)
    visualizer.plot_training_history(scenario=SCENARIO)
    visualizer.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=SCENARIO)
    visualizer.plot_test_metrics(result["metrics"], scenario=SCENARIO)
    visualizer.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=SCENARIO)