"""Regression test for Scenario 2 NBEATS epoch selection.

This script reloads the optimized checkpoints for ranks 1, 2, and 3,
rebuilds the trainer with the exact tuned hyperparameters, retrains every
fold from scratch twice, and checks that the fold best epochs, median epoch,
and final epoch match the saved checkpoint metadata.
"""

import sys
from pathlib import Path

# Add project root to sys.path to support direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch

from src.hpo.objective import build_walk_params
from src.hpo.trainer_factory import make_trainer
from src.data.walk_forward import WalkForwardSplitter


SCENARIO = 2
MODEL_TYPE = "nbeats"
SEED = 42
PRED_LEN = 3
EPOCHS = 3000
PATIENCE = 100
HOLDING_COST = 2.0
LEAD_TIME = 2
ORDERING_COST = 50_000.0
FORECAST_HORIZON = 4
CHECKPOINT_DIR = Path("checkpoints_optuna")


DECOMP_KEYS = {
    "decomposition_method",
    "seasonal_period",
    "stl_robust",
    "stl_seasonal",
    "stl_trend",
    "stl_low_pass",
    "trend_hidden_dim",
    "trend_n_layers",
    "seasonality_model",
    "aggregation_method",
    "learnable_aggregation",
    "hierarchical_decomposition",
}


def _load_checkpoint(rank: int) -> dict:
    path = CHECKPOINT_DIR / f"s2_nbeats_rank{rank}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_config(params: dict) -> dict:
    return {
        "pred_len": PRED_LEN,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "holding_cost": HOLDING_COST,
        "lead_time": LEAD_TIME,
        "ordering_cost": ORDERING_COST,
        "seed": SEED,
        "use_decomposition": any(key in params for key in DECOMP_KEYS),
        "decomposition_method": params.get("decomposition_method", "stl"),
        "seasonal_period": params.get("seasonal_period", 4),
        "stl_robust": params.get("stl_robust", True),
        "stl_seasonal": params.get("stl_seasonal", 7),
        "stl_trend": params.get("stl_trend", None),
        "stl_low_pass": params.get("stl_low_pass", None),
        "trend_hidden_dim": params.get("trend_hidden_dim", 32),
        "trend_n_layers": params.get("trend_n_layers", 1),
        "seasonality_model": params.get("seasonality_model", MODEL_TYPE),
        "aggregation_method": params.get("aggregation_method", "sum"),
        "learnable_aggregation": params.get("learnable_aggregation", False),
        "hierarchical_decomposition": params.get("hierarchical_decomposition", False),
    }


def _retrain_fold_epochs(trainer, params: dict, walk_params: dict) -> list[int]:
    splitter = WalkForwardSplitter(**walk_params)
    fold_best_epochs = []

    for split in splitter.get_splits():
        train_loader = trainer._make_loader(0, split["train_end"], params["batch_size"])
        val_loader = trainer._make_loader(split["train_end"], split["val_end"], params["batch_size"])

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        model = trainer._build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        _, best_epoch = trainer._train_one_fold(
            model,
            optimizer,
            train_loader,
            val_loader,
            fold_idx=split["fold"],
            max_epochs=trainer.epochs,
            verbose=False,
        )
        fold_best_epochs.append(int(best_epoch))

    return fold_best_epochs


def _assert_rank_checkpoint(rank: int) -> dict:
    checkpoint = _load_checkpoint(rank)
    params = checkpoint["params"].copy()
    config = _build_config(params)

    trainer = make_trainer(
        SCENARIO,
        params,
        config,
        seed=SEED,
        trial=None,
        model_type=MODEL_TYPE,
    )
    walk_params = build_walk_params(params, PRED_LEN)

    first_pass = _retrain_fold_epochs(trainer, params, walk_params)
    second_pass = _retrain_fold_epochs(trainer, params, walk_params)

    assert first_pass == second_pass, (
        f"rank {rank}: retraining is not stable across two passes: "
        f"pass1={first_pass}, pass2={second_pass}"
    )

    fold_best_epochs = first_pass
    expected_fold_epochs = [int(epoch) for epoch in checkpoint.get("fold_best_epochs", [])]

    assert fold_best_epochs == expected_fold_epochs, (
        f"rank {rank}: fold_best_epochs mismatch: "
        f"got={fold_best_epochs}, expected={expected_fold_epochs}"
    )

    median_best_epoch = int(round(float(np.median(fold_best_epochs)))) if fold_best_epochs else None
    expected_median = checkpoint.get("cv_median_best_epoch") or checkpoint.get("best_epoch")
    expected_final = checkpoint.get("final_train_epochs") or expected_median

    assert median_best_epoch == expected_median, (
        f"rank {rank}: median epoch mismatch: got={median_best_epoch}, expected={expected_median}"
    )

    final_result = trainer.train_and_test_with_best_hparams(
        walk_params,
        batch_size=params["batch_size"],
        verbose=False,
        fold_best_epochs=fold_best_epochs,
        fixed_epochs=median_best_epoch,
    )

    assert final_result["best_epoch"] == expected_final, (
        f"rank {rank}: final epoch mismatch: got={final_result['best_epoch']}, expected={expected_final}"
    )

    return {
        "rank": rank,
        "params": params,
        "fold_best_epochs": fold_best_epochs,
        "median_best_epoch": median_best_epoch,
        "final_epoch": final_result["best_epoch"],
        "expected_final": expected_final,
        "first_pass": first_pass,
        "second_pass": second_pass,
    }


def main() -> None:
    results = []
    for rank in (1, 2, 3):
        result = _assert_rank_checkpoint(rank)
        results.append(result)
        print(
            f"rank {rank}: first_pass={result['first_pass']} | "
            f"second_pass={result['second_pass']} | median={result['median_best_epoch']} | final={result['final_epoch']}"
        )

    print("\nAll NBEATS epoch checks passed for ranks 1, 2, and 3.")


if __name__ == "__main__":
    main()