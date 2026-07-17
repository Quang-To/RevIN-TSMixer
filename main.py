import ast
import re
from pathlib import Path

import torch
from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.evaluation import print_summary
from src.utils.visualization import TrainingVisualizer
import logging
import warnings

# ── Configuration ─────────────────────────────────────────────────────────────

SCENARIO   = 2
MODEL      = "nbeats"   # "tsmixer", "nbeats", or "nhits"
VAL_METRIC = "tc"      # "mape" for Scenario 1, "tc" for Scenario 2
SEED       = 42

# Model
SEQ_LENGTH = 12
PRED_LEN   = 3

# TSMixer specific
N_BLOCK    = 3
FF_DIM     = 64

# NBEATS specific
N_STACKS   = 2
N_LAYERS   = 3
LAYER_DIM  = 256

# NHITS specific
N_BLOCKS   = 1
HIDDEN_DIM = 128

# Common
DROPOUT    = 0.3

# Training
BATCH_SIZE = 4
LR         = 1e-5
EPOCHS     = 3000
PATIENCE   = 100

# Inventory cost
HOLDING_COST     = 2.0
LEAD_TIME        = 2
ORDERING_COST    = 50_000.0
FORECAST_HORIZON = 4

# ── Decomposition defaults ─────────────────────────────────────────────────────
FORCE_NO_DECOMPOSITION = False
FORCE_SUM_AGGREGATION = False
DECOMP_DEFAULTS = dict(
    decomposition_method      = "stl",
    seasonal_period           = 4,
    stl_robust                = True,
    stl_seasonal              = 7,
    stl_trend                 = None,
    stl_low_pass              = None,
    trend_hidden_dim          = 32,
    trend_n_layers            = 2,           
    seasonality_n_stacks      = 2,
    seasonality_n_blocks      = 1,
    seasonality_n_layers      = 4,
    seasonality_hidden_dim    = 256,
    seasonality_layer_dim     = 256,
    aggregation_method        = "weighted",
    learnable_aggregation     = False,
    hierarchical_decomposition= False,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(params: dict, checkpoint_params: dict, key: str, default):
    return checkpoint_params.get(key, params.get(key, default))


def _parse_summary_checkpoint(summary_path: Path) -> dict:
    text = summary_path.read_text(encoding="utf-8")
    data = {
        "params": {},
        "fold_best_epochs": [],
        "rank": None,
        "scenario": None,
        "model_type": None,
        "val_metric": None,
        "best_epoch": None,
        "cv_median_best_epoch": None,
        "median_best_epoch": None,
    }

    header_match = re.search(r"Scenario\s+(\d+)\s+—\s+Model\s+(\w+)\s+—\s+Rank\s+(\d+)", text)
    if header_match:
        data["scenario"] = int(header_match.group(1))
        data["model_type"] = header_match.group(2).lower()
        data["rank"] = int(header_match.group(3))

    val_match = re.search(r"Val Metric\s*:\s*(\w+)", text)
    if val_match:
        data["val_metric"] = val_match.group(1).lower()

    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("── Hyperparameters"):
            section = "params"
            continue
        if line.startswith("── Fold best epochs"):
            section = "fold_epochs"
            continue
        if line.startswith("── Validation metric"):
            section = None
            continue

        if section == "params" and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value
            data["params"][key] = parsed_value

        elif section == "fold_epochs" and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value

            if key == "epochs" and isinstance(parsed_value, list):
                data["fold_best_epochs"] = parsed_value
            elif key in {"median", "final epochs"} and parsed_value is not None:
                try:
                    epoch_value = int(parsed_value)
                except (TypeError, ValueError):
                    continue
                data["cv_median_best_epoch"] = epoch_value
                data["median_best_epoch"] = epoch_value
                data["best_epoch"] = epoch_value

    if data["fold_best_epochs"] and data["cv_median_best_epoch"] is None:
        import numpy as np

        median_epoch = int(round(float(np.median(data["fold_best_epochs"]))))
        data["cv_median_best_epoch"] = median_epoch
        data["median_best_epoch"] = median_epoch
        data["best_epoch"] = median_epoch

    data["decomp_params"] = {
        key: data["params"][key]
        for key in DECOMP_DEFAULTS
        if key in data["params"]
    }
    return data


def _summary_has_epochs(summary: dict) -> bool:
    return bool(summary.get("fold_best_epochs")) or summary.get("best_epoch") is not None


def _checkpoint_has_epochs(checkpoint: dict) -> bool:
    return bool(checkpoint.get("fold_best_epochs")) or checkpoint.get("best_epoch") is not None


def _build_summary_text_from_checkpoint(checkpoint: dict, scenario: int, model_name: str, rank: int) -> str:
    import numpy as np

    def _fmt(value):
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value) if value is not None else "N/A"

    params = checkpoint.get("params", {}) if isinstance(checkpoint, dict) else {}
    fold_best_epochs = [int(epoch) for epoch in checkpoint.get("fold_best_epochs", []) if epoch is not None]
    best_epoch = checkpoint.get("best_epoch")
    if not fold_best_epochs and best_epoch is not None:
        fold_best_epochs = [int(best_epoch)]

    median_best_epoch = checkpoint.get("cv_median_best_epoch")
    if median_best_epoch is None and fold_best_epochs:
        median_best_epoch = int(round(float(np.median(fold_best_epochs))))

    final_train_epochs = checkpoint.get("final_train_epochs", median_best_epoch)
    val_metric = checkpoint.get("val_metric", checkpoint.get("value"))
    metric_label = "MAPE (%)" if scenario == 1 else "TC_min"
    fold_mean = checkpoint.get("cv_mean", val_metric)
    fold_std = checkpoint.get("cv_std", 0.0)
    fold_min = checkpoint.get("cv_min", val_metric)
    fold_max = checkpoint.get("cv_max", val_metric)

    lines = [
        f"Scenario {scenario} — Model {model_name.upper()} — Rank {rank}",
        "=" * 50,
        "",
        "── Loss Configuration ──────────────────────────",
        f"  {'Val Metric':<12}: {_fmt(val_metric)}",
        f"  {'Model':<12}: {model_name.upper()}",
        "",
        "── Hyperparameters ─────────────────────────────",
        *[f"  {k:<12}: {v}" for k, v in params.items()],
        "",
        "── Fold best epochs ────────────────────────────",
        f"  {'epochs':<12}: {fold_best_epochs}",
        f"  {'median':<12}: {_fmt(median_best_epoch)}",
        f"  {'final epochs':<12}: {_fmt(final_train_epochs)}",
        "",
        "── Validation metric (CV mean) ──────────────────",
        f"  {metric_label:<12}: {_fmt(val_metric)}",
        "",
        "── CV fold statistics ───────────────────────────",
        f"  {'fold mean':<12}: {_fmt(fold_mean)}",
        f"  {'fold std':<12}: {_fmt(fold_std)}",
        f"  {'fold min':<12}: {_fmt(fold_min)}",
        f"  {'fold max':<12}: {_fmt(fold_max)}",
    ]

    final_result = checkpoint.get("final_result") if isinstance(checkpoint, dict) else None
    if final_result is not None:
        final_metrics = final_result.get("metrics", {}) if isinstance(final_result, dict) else {}
        lines.extend([
            "",
            "── Final evaluation ───────────────────────────",
            f"  {'final_epochs':<12}: {final_result.get('best_epoch', 'N/A')}",
            f"  {'final_Tc_min':<12}: {_fmt(final_metrics.get('TC_min'))}",
            f"  {'final_MAPE':<12}: {_fmt(final_metrics.get('MAPE'))}",
        ])

    return "\n".join(lines)


def _repair_summary_from_checkpoint(base_path: Path, checkpoint: dict, scenario: int, model_name: str, rank: int) -> None:
    txt_path = base_path.with_suffix(".txt")
    if not isinstance(checkpoint, dict) or not _checkpoint_has_epochs(checkpoint):
        return

    if txt_path.exists():
        try:
            summary = _parse_summary_checkpoint(txt_path)
        except Exception:
            summary = None
        if summary is not None and _summary_has_epochs(summary):
            return

    txt_path.write_text(_build_summary_text_from_checkpoint(checkpoint, scenario, model_name, rank), encoding="utf-8")


def load_checkpoint_or_summary(scenario: int, model_name: str, rank: int = 1) -> dict:
    base_path = Path("checkpoints_optuna") / f"s{scenario}_{model_name}_rank{rank}"
    pt_path = base_path.with_suffix(".pt")
    if pt_path.exists():
        try:
            checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = None
        if isinstance(checkpoint, dict):
            _repair_summary_from_checkpoint(base_path, checkpoint, scenario, model_name, rank)
            return checkpoint

    txt_path = base_path.with_suffix(".txt")
    if txt_path.exists():
        return _parse_summary_checkpoint(txt_path)

    raise FileNotFoundError(
        f"No checkpoint found for scenario={scenario}, model={model_name!r}, rank={rank}. "
        f"Looked for {pt_path.name} and {txt_path.name}."
    )


def load_best_checkpoint_or_summary(scenario: int, model_name: str, ranks=(1, 2, 3)) -> dict:
    """Load the best available checkpoint across the given ranks.

    The selected checkpoint is the one with the smallest validation metric.
    This is safer than assuming rank 1 is still the best file on disk,
    especially after resume/rebuild workflows.
    """
    candidates = []
    for rank in ranks:
        try:
            checkpoint = load_checkpoint_or_summary(scenario, model_name, rank=rank)
        except FileNotFoundError:
            continue

        val_metric = checkpoint.get("val_metric", None)
        if val_metric is None:
            continue
        candidates.append((float(val_metric), rank, checkpoint))

    if not candidates:
        raise FileNotFoundError(
            f"No usable checkpoint found for scenario={scenario}, model={model_name!r}. "
            f"Checked ranks={tuple(ranks)}."
        )

    candidates.sort(key=lambda item: item[0])
    best_val, best_rank, best_checkpoint = candidates[0]
    logging.info(f"Selected checkpoint rank={best_rank} with val_metric={best_val:.4f}")
    return best_checkpoint

def build_trainer(
    model_name: str,
    best_params: dict,
    decomp_params: dict,
    TrainerClass,
    visualizer,
):
    common = dict(
        pred_len     = best_params.get("pred_len", best_params.get("pred_length", PRED_LEN)),
        batch_size   = best_params.get("batch_size", BATCH_SIZE),
        lr           = best_params.get("lr", LR),
        epochs       = best_params.get("epochs", EPOCHS),
        patience     = best_params.get("patience", PATIENCE),
        holding_cost = best_params.get("holding_cost", HOLDING_COST),
        ordering_cost= best_params.get("ordering_cost", ORDERING_COST),
        lead_time    = best_params.get("lead_time", LEAD_TIME),
        seed         = best_params.get("seed", SEED),
        visualizer   = visualizer,
    )

    decomp = dict(
        use_decomposition          = best_params.get("use_decomposition", False),
        decomposition_method       = decomp_params["decomposition_method"],
        seasonal_period            = decomp_params["seasonal_period"],
        stl_robust                 = decomp_params["stl_robust"],
        stl_seasonal               = decomp_params["stl_seasonal"],
        stl_trend                  = decomp_params["stl_trend"],
        stl_low_pass               = decomp_params["stl_low_pass"],
        trend_hidden_dim           = decomp_params["trend_hidden_dim"],
        trend_n_layers             = decomp_params["trend_n_layers"],
        aggregation_method         = decomp_params["aggregation_method"],
        learnable_aggregation      = decomp_params["learnable_aggregation"],
        hierarchical_decomposition = decomp_params["hierarchical_decomposition"],
        seasonality_model          = best_params.get("seasonality_model", best_params.get("seasonality_model_type", "tsmixer")),
    )

    if FORCE_SUM_AGGREGATION and decomp["use_decomposition"]:
        decomp["aggregation_method"] = "sum"
        decomp["learnable_aggregation"] = False

    if model_name == "tsmixer":
        model_kwargs = dict(
            seq_length = best_params.get("seq_length", SEQ_LENGTH),
            ff_dim     = best_params.get("ff_dim", FF_DIM),
            n_block    = best_params.get("n_block", N_BLOCK),
            dropout    = best_params.get("dropout", DROPOUT),
            model_type = "tsmixer",
        )
        all_kwargs = {**common, **decomp, **model_kwargs}
        return TrainerClass(**all_kwargs)

    elif model_name == "nbeats":
        model_kwargs = dict(
            seq_length = best_params.get("seq_length", SEQ_LENGTH),
            n_stacks   = best_params.get("n_stacks", N_STACKS),
            n_layers   = best_params.get("n_layers", N_LAYERS),
            layer_dim  = best_params.get("layer_dim", LAYER_DIM),
            dropout    = best_params.get("dropout", DROPOUT),
            model_type = "nbeats",
        )
        if "seasonality_n_stacks" in decomp_params:
            model_kwargs["n_stacks"] = decomp_params["seasonality_n_stacks"]
        if "seasonality_n_layers" in decomp_params:
            model_kwargs["n_layers"] = decomp_params["seasonality_n_layers"]
        if "seasonality_layer_dim" in decomp_params:
            model_kwargs["layer_dim"] = decomp_params["seasonality_layer_dim"]
        all_kwargs = {**common, **decomp, **model_kwargs}
        return TrainerClass(**all_kwargs)

    elif model_name == "nhits":
        model_kwargs = dict(
            seq_length = best_params.get("seq_length", SEQ_LENGTH),
            n_stacks   = best_params.get("n_stacks", N_STACKS),
            n_blocks   = best_params.get("n_blocks", N_BLOCKS),
            n_layers   = best_params.get("n_layers", N_LAYERS),
            hidden_dim = best_params.get("hidden_dim", HIDDEN_DIM),
            dropout    = best_params.get("dropout", DROPOUT),
            model_type = "nhits",
        )
        if "seasonality_n_stacks" in decomp_params:
            model_kwargs["n_stacks"] = decomp_params["seasonality_n_stacks"]
        if "seasonality_n_blocks" in decomp_params:
            model_kwargs["n_blocks"] = decomp_params["seasonality_n_blocks"]
        if "seasonality_n_layers" in decomp_params:
            model_kwargs["n_layers"] = decomp_params["seasonality_n_layers"]
        if "seasonality_hidden_dim" in decomp_params:
            model_kwargs["hidden_dim"] = decomp_params["seasonality_hidden_dim"]
        if "seasonality_layer_dim" in decomp_params:
            model_kwargs["layer_dim"] = decomp_params["seasonality_layer_dim"]
        all_kwargs = {**common, **decomp, **model_kwargs}
        return TrainerClass(**all_kwargs)

    else:
        raise ValueError(f"Unknown MODEL: {model_name!r}. Must be 'tsmixer', 'nbeats', or 'nhits'")


# ── Pipeline (concise) ───────────────────────────────────────────────────────

def main():
    # Suppress UserWarnings (including noisy matplotlib warnings) and configure logging to show important info
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # No global print overrides; decomposition debug messages use logging.debug now

    ckpt = load_best_checkpoint_or_summary(SCENARIO, MODEL, ranks=(1, 2, 3))
    best_params = ckpt.get("params", {}).copy()
    best_params["use_decomposition"] = (
        any(k in best_params for k in DECOMP_DEFAULTS) or bool(best_params.get("use_decomposition", False))
    )
    if FORCE_NO_DECOMPOSITION:
        best_params["use_decomposition"] = False

    median_best_epoch = ckpt.get("cv_median_best_epoch") or ckpt.get("median_best_epoch") or ckpt.get("best_epoch")
    fold_best_epochs = ckpt.get("fold_best_epochs", [])
    val_metric = ckpt.get("val_metric")

    for k in ("val_metric", "val_metric_type"):
        best_params.pop(k, None)

    logging.info(f"Loaded checkpoint: rank={ckpt.get('rank')} scenario={ckpt.get('scenario')} model={ckpt.get('model_type')} val_metric={val_metric}")
    logging.info(f"Decomposition enabled: {best_params['use_decomposition']}")

    fixed_epochs = median_best_epoch
    if fixed_epochs is None and fold_best_epochs:
        import numpy as np
        fixed_epochs = int(round(float(np.median(fold_best_epochs))))
        logging.info(f"Fixed final epochs: {fixed_epochs}")

    decomp_params = {k: _get(best_params, ckpt.get("decomp_params", {}), k, default) for k, default in DECOMP_DEFAULTS.items()}
    logging.info("Decomposition params being used:")
    for k, v in decomp_params.items():
        src = "checkpoint.decomp_params" if k in ckpt.get("decomp_params", {}) else ("best_params" if k in best_params else "default")
        logging.info(f"  {k:<28} = {v!r:>10}   [{src}]")

    TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer
    trainer = build_trainer(MODEL, best_params, decomp_params, TrainerClass, TrainingVisualizer(save_dir="results"))

    walk_params = dict(
        seq_length=best_params.get("seq_length", SEQ_LENGTH),
        pred_length=best_params.get("pred_length", best_params.get("pred_len", PRED_LEN)),
        forecast_horizon=best_params.get("forecast_horizon", FORECAST_HORIZON),
        train_ratio=best_params.get("train_ratio", 0.6),
        val_size=best_params.get("val_size", 21),
        test_size=best_params.get("test_size", 21),
        step=best_params.get("step", 3),
    )

    run_result = trainer.train_and_test_with_best_hparams(
        walk_params,
        batch_size=best_params.get("batch_size", BATCH_SIZE),
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

    print_summary(result, scenario=SCENARIO, seed=SEED)
    viz = trainer.visualizer
    viz.plot_training_history(scenario=SCENARIO)
    viz.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=SCENARIO)
    viz.plot_test_metrics(result["metrics"], scenario=SCENARIO)
    viz.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=SCENARIO)
    if result.get("decomp_components") is not None:
        viz.plot_decomposition_diagnostics(result["decomp_components"], result["trues"], scenario=SCENARIO)


if __name__ == "__main__":
    main()