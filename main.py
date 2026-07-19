# main.py

import logging
import warnings
import numpy as np

import config
from src.checkpoints import load_best_checkpoint_or_summary
from src.models.factory import build_trainer
from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.visualization import TrainingVisualizer
from src.utils.reporting import print_run_summary


def main():
    # Suppress UserWarnings (including matplotlib warnings) and configure logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ckpt = load_best_checkpoint_or_summary(config.SCENARIO, config.MODEL, ranks=(1, 2, 3))
    best_params = ckpt.get("params", {}).copy()
    best_params["use_decomposition"] = (
        any(k in best_params for k in config.DECOMP_DEFAULTS) or bool(best_params.get("use_decomposition", False))
    )
    best_params["use_log_return"] = best_params.get("use_log_return", config.USE_LOG_RETURN)
    if config.FORCE_NO_DECOMPOSITION:
        best_params["use_decomposition"] = False

    median_best_epoch = ckpt.get("cv_median_best_epoch") or ckpt.get("median_best_epoch") or ckpt.get("best_epoch")
    fold_best_epochs = ckpt.get("fold_best_epochs", [])
    val_metric = ckpt.get("val_metric")

    for k in ("val_metric", "val_metric_type"):
        best_params.pop(k, None)

    # Resolve decomposition parameters
    from src.checkpoints.parsing import _get
    decomp_params = {
        k: _get(best_params, ckpt.get("decomp_params", {}), k, default)
        for k, default in config.DECOMP_DEFAULTS.items()
    }
    
    logging.info(f"Loaded checkpoint: rank={ckpt.get('rank')} model={ckpt.get('model_type', config.MODEL).upper()}")
    logging.info(f"Decomposition: {decomp_params['decomposition_method'].upper()}")
    logging.info(f"seasonal_period={decomp_params['seasonal_period']}")
    if decomp_params['decomposition_method'].lower() == "stl":
        logging.info(f"stl_seasonal={decomp_params['stl_seasonal']}")
        logging.info(f"stl_trend={decomp_params['stl_trend']}")

    # Print model parameters
    model_keys = {
        "tsmixer": ["seq_length", "n_block", "dropout", "batch_size", "ff_dim", "lr"],
        "nbeats": ["seq_length", "n_stacks", "n_layers", "layer_dim", "dropout", "batch_size", "lr"],
        "nhits": ["seq_length", "n_stacks", "n_blocks", "n_layers", "hidden_dim", "dropout", "batch_size", "lr"],
    }.get(config.MODEL.lower(), [])
    params_str = ", ".join(
        f"{k}={best_params.get(k, ckpt.get('params', {}).get(k))}"
        for k in model_keys
        if best_params.get(k) is not None or ckpt.get('params', {}).get(k) is not None
    )
    logging.info(f"Parameters: {params_str}")

    fixed_epochs = median_best_epoch
    if fixed_epochs is None and fold_best_epochs:
        fixed_epochs = int(round(float(np.median(fold_best_epochs))))
        logging.debug(f"Fixed final epochs: {fixed_epochs}")

    TrainerClass = Scenario1Trainer if config.SCENARIO == 1 else Scenario2Trainer
    trainer = build_trainer(
        config.MODEL,
        best_params,
        decomp_params,
        TrainerClass,
        TrainingVisualizer(save_dir="results"),
        ckpt_decomp_params=ckpt.get("decomp_params", {})
    )

    walk_params = dict(
        seq_length=best_params.get("seq_length", config.SEQ_LENGTH),
        pred_length=best_params.get("pred_length", best_params.get("pred_len", config.PRED_LEN)),
        forecast_horizon=best_params.get("forecast_horizon", config.FORECAST_HORIZON),
        train_ratio=best_params.get("train_ratio", 0.6),
        val_size=best_params.get("val_size", 21),
        test_size=best_params.get("test_size", 21),
        step=best_params.get("step", 3),
    )

    run_result = trainer.train_and_test_with_best_hparams(
        walk_params,
        batch_size=best_params.get("batch_size", config.BATCH_SIZE),
        verbose=True,
        n_epochs=trainer.epochs,
        patience=trainer.patience,
        fold_best_epochs=fold_best_epochs,
        fixed_epochs=fixed_epochs,
    )

    result = {
        "preds": run_result["test_pred"],
        "trues": run_result["test_true"],
        "indices": run_result.get("test_indices", []),
        "decomp_components": run_result.get("decomp_components"),
        "metrics": run_result["metrics"],
        "best_epoch": run_result.get("best_epoch"),
        "median_best_epoch": median_best_epoch,
        "fold_best_epochs": fold_best_epochs,
        "val_metric": val_metric,
        "rank": ckpt.get("rank"),
        "scenario": ckpt.get("scenario"),
        "model": ckpt.get("model_type"),
    }

    # Print run summary
    print_run_summary(result, config.SCENARIO, best_params.get("seq_length", config.SEQ_LENGTH))

    # Plot visualizer
    viz = trainer.visualizer
    viz.plot_training_history(scenario=config.SCENARIO)
    viz.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=config.SCENARIO)
    viz.plot_test_metrics(result["metrics"], scenario=config.SCENARIO)
    viz.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=config.SCENARIO)
    if result.get("decomp_components") is not None:
        viz.plot_decomposition_diagnostics(result["decomp_components"], result["trues"], scenario=config.SCENARIO)
        viz.plot_time_series_decomposition(
            result["decomp_components"],
            scenario=config.SCENARIO,
            seq_length=best_params.get("seq_length", config.SEQ_LENGTH)
        )


if __name__ == "__main__":
    main()