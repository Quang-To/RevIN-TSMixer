from src.hpo.config import DEFAULT_SPLIT_PARAMS


def sample_params(trial, model_type: str = "tsmixer") -> dict:
    """Lấy mẫu siêu tham số chuẩn xác từ không gian tìm kiếm Optuna (liên tục + log-scale)."""
    model_name = model_type.lower()
    p = {
        "seq_length": trial.suggest_categorical("lookback", [6, 9, 12, 18, 24]),
        "lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "revin_affine": trial.suggest_categorical("revin_affine", [True, False]),
        "hidden_size": trial.suggest_int("hidden_size", 64, 512, step=64),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    # Thêm alias để tương thích
    p["lookback"] = p["seq_length"]
    p["learning_rate"] = p["lr"]
    p["ff_dim"] = p["hidden_size"]
    p["layer_dim"] = p["hidden_size"]
    p["hidden_dim"] = p["hidden_size"]
    p["n_block"] = p["n_layers"]

    if model_name == "nbeats":
        p["n_stacks"] = trial.suggest_int("n_stacks", 1, 3)
        p["n_blocks"] = trial.suggest_int("n_blocks", 1, 4)
    elif model_name == "nhits":
        p["n_stacks"] = trial.suggest_int("n_stacks", 1, 3)
        p["n_blocks"] = trial.suggest_int("n_blocks", 1, 4)

    return p


def build_split_params(params: dict, pred_length: int = 1) -> dict:
    """Build fixed 7:1:2 split parameters for training."""
    return {
        **DEFAULT_SPLIT_PARAMS,
        "seq_length": params.get("seq_length", params.get("lookback", 24)),
        "pred_length": pred_length,
    }

build_walk_params = build_split_params


def evaluate(trainer, split_params: dict, params: dict) -> tuple:
    """Evaluate candidate trial on validation set."""
    result = trainer.evaluate_for_optuna(
        split_params,
        batch_size=params["batch_size"],
        verbose=False,
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def config_dict_from_obj(obj) -> dict:
    """Extract trainer config dict from an OptunaOptimizer instance."""
    try:
        import config
        default_shortage = config.SHORTAGE_COST
    except ImportError:
        default_shortage = 30.0

    return {
        "pred_len":      getattr(obj, "pred_len", 1),
        "epochs":        getattr(obj, "epochs", 300),
        "patience":      getattr(obj, "patience", 30),
        "holding_cost":  getattr(obj, "holding_cost", 4.0),
        "shortage_cost": getattr(obj, "shortage_cost", default_shortage),
        "lead_time":     getattr(obj, "lead_time", 1),
        "ordering_cost": getattr(obj, "ordering_cost", 50000.0),
        "val_metric":    getattr(obj, "val_metric", "pinball"),
        "seed":          int(getattr(obj, "seed", 42)),
        "use_log_return": bool(getattr(obj, "use_log_return", False)),
        "data_file":     getattr(obj, "data_file", None),
    }
