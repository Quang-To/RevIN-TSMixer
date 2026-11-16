"""Train and evaluate a selected forecasting model.

The entry point intentionally keeps orchestration here and model logic in
``src``. This makes the default experiment reproducible while allowing a
reviewer to override the dataset, checkpoint directory, and runtime budget.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import config

from src.checkpoints import load_best_checkpoint_or_summary
from src.trainers import Scenario1Trainer, Scenario2Trainer
from src.models.factory import build_trainer
from src.utils.reporting import print_run_summary
from src.utils.visualization import TrainingVisualizer

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints_optuna"
DEFAULT_OUTPUT_DIR = ROOT / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a probabilistic demand forecasting model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", type=int, choices=(1, 2), default=config.SCENARIO)
    parser.add_argument(
        "--model",
        choices=("tsmixer", "nbeats", "nhits"),
        default=config.MODEL.lower(),
    )
    parser.add_argument(
        "--data-file",
        default=config.DATA_FILE,
        help="CSV filename in data/raw_data or an absolute CSV path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing Optuna rank checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots and generated evaluation artifacts.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="1,2,3",
        help="Comma-separated checkpoint ranks to compare before selecting the best.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the maximum training epochs from the selected checkpoint.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early-stopping patience from the selected checkpoint.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Train and evaluate without generating visualization files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the checkpoint and print its parameters without training.",
    )
    return parser


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be a comma-separated list of integers") from exc
    if not ranks or any(rank < 1 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must contain positive integers")
    return ranks


def _resolve_data_file(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Data file does not exist: {path}")
        return str(path)

    candidate = ROOT / "data" / "raw_data" / path
    if not candidate.exists():
        raise FileNotFoundError(
            f"Data file {value!r} was not found in {candidate.parent}. "
            "Pass --data-file with a valid filename or absolute path."
        )
    return value


def _checkpoint_data_tag(data_file: str) -> str | None:
    """Return the legacy checkpoint data tag only for non-default datasets."""
    if Path(data_file).name == Path(config.DATA_FILE).name:
        return None
    return data_file


def _load_params(args: argparse.Namespace) -> dict:
    ranks = _parse_ranks(args.ranks)
    try:
        checkpoint = load_best_checkpoint_or_summary(
            scenario=args.scenario,
            model_name=args.model,
            ranks=ranks,
            data_file=_checkpoint_data_tag(args.data_file),
            checkpoint_dir=args.checkpoint_dir,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nRun the corresponding HPO command first or point "
            "to the correct --checkpoint-dir."
        ) from exc

    params = checkpoint.get("params", {}).copy()
    if not isinstance(params, dict):
        raise ValueError("Selected checkpoint contains an invalid 'params' mapping.")

    params.setdefault("use_log_return", config.USE_LOG_RETURN)
    params["data_file"] = args.data_file
    params["seed"] = args.seed
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.patience is not None:
        params["patience"] = args.patience

    return {
        "checkpoint": checkpoint,
        "params": params,
        "ranks": ranks,
    }


def run(args: argparse.Namespace) -> dict:
    data_file = _resolve_data_file(args.data_file)
    args.data_file = data_file
    loaded = _load_params(args)
    checkpoint = loaded["checkpoint"]
    best_params = loaded["params"]

    logging.info(
        "Selected checkpoint: model=%s rank=%s val_metric=%s",
        args.model.upper(),
        checkpoint.get("rank", "unknown"),
        checkpoint.get("val_metric", "unknown"),
    )

    if args.dry_run:
        print("Resolved configuration:")
        for key in sorted(best_params):
            print(f"  {key}: {best_params[key]}")
        return {"checkpoint": checkpoint, "params": best_params}

    trainer_class = Scenario1Trainer if args.scenario == 1 else Scenario2Trainer
    visualizer = None if args.no_plots else TrainingVisualizer(
        save_dir=args.output_dir,
        data_file=args.data_file,
    )
    trainer = build_trainer(
        model_name=args.model,
        best_params=best_params,
        TrainerClass=trainer_class,
        visualizer=visualizer,
    )

    n = trainer.data_dict["n"]
    train_end = trainer.data_dict["train_end"]
    val_end = trainer.data_dict["val_end"]
    split_params = {
        "seq_length": best_params.get("seq_length", config.SEQ_LENGTH),
        "pred_length": best_params.get("pred_length", config.PRED_LEN),
        "forecast_horizon": config.FORECAST_HORIZON,
        "train_size": train_end,
        "val_size": val_end - train_end,
        "test_size": n - val_end,
    }

    logging.info(
        "Training %s on scenario %s: train=[0:%d], val=[%d:%d], test=[%d:%d]",
        args.model.upper(),
        args.scenario,
        train_end,
        train_end,
        val_end,
        val_end,
        n,
    )
    run_result = trainer.train_with_early_stopping_and_test(
        split_params,
        batch_size=best_params.get("batch_size", config.BATCH_SIZE),
        verbose=True,
        max_epochs=best_params.get("epochs", trainer.epochs),
        patience=best_params.get("patience", trainer.patience),
    )

    result = {
        "preds": run_result["test_pred"],
        "trues": run_result["test_true"],
        "indices": run_result.get("test_indices", []),
        "quantiles": run_result.get("test_quantiles"),
        "metrics": run_result["metrics"],
        "best_epoch": run_result.get("best_epoch"),
        "median_best_epoch": run_result.get("best_epoch"),
        "mean_best_epoch": run_result.get("best_epoch"),
        "fold_best_epochs": [run_result.get("best_epoch")],
        "val_metric": checkpoint.get("val_metric"),
        "rank": checkpoint.get("rank"),
        "scenario": args.scenario,
        "model": args.model,
    }
    print_run_summary(result, args.scenario, best_params.get("seq_length", config.SEQ_LENGTH))

    if visualizer is not None:
        visualizer.plot_training_history(scenario=args.scenario)
        visualizer.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=args.scenario)
        visualizer.plot_test_metrics(result["metrics"], scenario=args.scenario)
        visualizer.plot_comparison_with_baseline(
            result["preds"], result["trues"], scenario=args.scenario
        )
        visualizer.plot_fan_chart(
            result["quantiles"],
            result["trues"],
            result["indices"],
            scenario=args.scenario,
        )
        visualizer.plot_calibration_curve(
            result["quantiles"], result["trues"], scenario=args.scenario
        )
        visualizer.plot_expected_cost_curve(
            result["quantiles"],
            result["trues"],
            holding_cost=config.HOLDING_COST,
            shortage_cost=config.SHORTAGE_COST,
            scenario=args.scenario,
        )
        visualizer.plot_eda()
        visualizer.update_model_comparison(args.model, result["metrics"], scenario=args.scenario)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
