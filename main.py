import torch
from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.evaluation import print_summary
from src.utils.visualization import TrainingVisualizer

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
# Các giá trị này là fallback nếu checkpoint không chứa decomp params.
DECOMP_DEFAULTS = dict(
    decomposition_method      = "savgol",
    seasonal_period           = 4,
    trend_hidden_dim          = 32,
    trend_n_layers            = 2,           # tên chuẩn khớp với TrendBranch.__init__
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
    """
    Ưu tiên: checkpoint_params > params (best_params từ optuna) > default.
    Cho phép decomp params được tune ở lần chạy optuna sau này tự động
    được nhận ra mà không cần sửa code.
    """
    return checkpoint_params.get(key, params.get(key, default))


def build_trainer(
    model_name: str,
    best_params: dict,
    decomp_params: dict,
    TrainerClass,
    visualizer,
):
    """
    Khởi tạo trainer tương ứng với model_name, merge tất cả params.
    decomp_params đã được resolve từ checkpoint + DECOMP_DEFAULTS trước khi
    truyền vào đây.
    """

    # Tham số chung cho mọi trainer
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

    # Tham số decomposition — luôn truyền vào, trainer tự quyết định dùng hay không
    # Decomposition args that don't conflict with model-specific kwargs
    decomp = dict(
        use_decomposition          = best_params.get("use_decomposition", False),
        decomposition_method       = decomp_params["decomposition_method"],
        seasonal_period            = decomp_params["seasonal_period"],
        trend_hidden_dim           = decomp_params["trend_hidden_dim"],
        trend_n_layers             = decomp_params["trend_n_layers"],
        aggregation_method         = decomp_params["aggregation_method"],
        learnable_aggregation      = decomp_params["learnable_aggregation"],
        hierarchical_decomposition = decomp_params["hierarchical_decomposition"],
        seasonality_model          = best_params.get("seasonality_model", best_params.get("seasonality_model_type", "tsmixer")),
    )

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
        # Allow decomp-specific seasonality params to override model kwargs if present
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
        # Allow decomp-specific seasonality params to override model kwargs
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


# ── Pipeline ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*30} FINAL TEST {'='*30}")

    # 1. Load checkpoint
    checkpoint = torch.load(f"checkpoints_optuna/s{SCENARIO}_{MODEL}_rank1.pt")
    best_params      = checkpoint["params"].copy()
    has_decomp_params = any(key in best_params for key in DECOMP_DEFAULTS)
    best_params["use_decomposition"] = has_decomp_params or bool(best_params.get("use_decomposition", False))
    best_epoch       = checkpoint.get("best_epoch", None)
    fold_best_epochs = checkpoint.get("fold_best_epochs", [])
    val_metric       = checkpoint.get("val_metric", None)
    rank             = checkpoint.get("rank", None)
    scenario_ckpt    = checkpoint.get("scenario", None)
    model_ckpt       = checkpoint.get("model_type", "tsmixer")

    # Xóa các key meta không phải hyperparameter
    for k in ("val_metric", "val_metric_type"):
        best_params.pop(k, None)

    print(
        f"Loaded checkpoint: rank={rank}, scenario={scenario_ckpt}, "
        f"model={model_ckpt}, val_metric={val_metric}, "
        f"best_epoch={best_epoch}, fold_best_epochs={fold_best_epochs}"
    )
    print(f"Decomposition enabled : {best_params['use_decomposition']}")

    decomp_params = {
        key: _get(best_params, checkpoint.get("decomp_params", {}), key, default)
        for key, default in DECOMP_DEFAULTS.items()
    }

    print("\nDecomposition params being used:")
    for k, v in decomp_params.items():
        source = (
            "checkpoint.decomp_params" if k in checkpoint.get("decomp_params", {})
            else "best_params"          if k in best_params
            else "default"
        )
        print(f"  {k:<34} = {v!r:>10}   [{source}]")

    # 3. Build trainer
    TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer
    visualizer   = TrainingVisualizer(save_dir="results")

    trainer = build_trainer(
        model_name   = MODEL,
        best_params  = best_params,
        decomp_params= decomp_params,
        TrainerClass = TrainerClass,
        visualizer   = visualizer,
    )

    # 4. Train & evaluate
    walk_params = dict(
        seq_length       = best_params.get("seq_length", SEQ_LENGTH),
        pred_length      = best_params.get("pred_length", best_params.get("pred_len", PRED_LEN)),
        forecast_horizon = best_params.get("forecast_horizon", FORECAST_HORIZON),
        train_ratio      = best_params.get("train_ratio", 0.6),
        val_size         = best_params.get("val_size", 21),
        test_size        = best_params.get("test_size", 21),
        step             = best_params.get("step", 3),
    )

    run_result = trainer.train_and_test_with_best_hparams(
        walk_params,
        batch_size = best_params.get("batch_size", BATCH_SIZE),
        verbose    = True,
        n_epochs   = trainer.epochs,
        patience   = trainer.patience,
    )

    result = {
        "preds"            : run_result["test_pred"],
        "trues"            : run_result["test_true"],
        "indices"          : run_result.get("test_indices", []),
        "decomp_components": run_result.get("decomp_components"),
        "metrics"          : run_result["metrics"],
        "best_epoch"       : run_result.get("best_epoch", None),
        "fold_best_epochs" : fold_best_epochs,
        "val_metric"       : val_metric,
        "rank"             : rank,
        "scenario"         : scenario_ckpt,
        "model"            : model_ckpt,
    }

    print_summary(result, scenario=SCENARIO, seed=SEED)
    visualizer.plot_training_history(scenario=SCENARIO)
    visualizer.plot_predictions_vs_actual(result["preds"], result["trues"], scenario=SCENARIO)
    visualizer.plot_test_metrics(result["metrics"], scenario=SCENARIO)
    visualizer.plot_comparison_with_baseline(result["preds"], result["trues"], scenario=SCENARIO)
    if result.get("decomp_components") is not None:
        visualizer.plot_decomposition_diagnostics(
            result["decomp_components"],
            result["trues"],
            scenario=SCENARIO,
        )