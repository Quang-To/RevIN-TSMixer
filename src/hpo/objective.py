from src.hpo.config import (
    DEFAULT_WALK_PARAMS,
    SEARCH_SPACE_TSMIXER,
    SEARCH_SPACE_NBEATS,
    SEARCH_SPACE_NHITS,
    SEARCH_SPACE_DECOMP,
)
from src.hpo.trainer_factory import make_trainer

def sample_params(trial, model_type="tsmixer", use_decomposition: bool = False):
    """Sample hyperparameters from Optuna trial."""
    if model_type == "tsmixer":
        base_space = SEARCH_SPACE_TSMIXER
    elif model_type == "nbeats":
        base_space = SEARCH_SPACE_NBEATS
    elif model_type == "nhits":
        base_space = SEARCH_SPACE_NHITS
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    search_space = dict(base_space)
    if use_decomposition:
        search_space.update(SEARCH_SPACE_DECOMP)
    
    return {k: trial.suggest_categorical(k, v) for k, v in search_space.items()}

def build_walk_params(params, pred_length):
    """Build walk-forward parameters for training."""
    return {**DEFAULT_WALK_PARAMS, "seq_length": params["seq_length"], "pred_length": pred_length}

def evaluate(trainer, walk_params, params):
    """Evaluate trainer and return (loss, metrics) tuple."""
    result = trainer.crossval_loss_for_optuna(
        walk_params,
        batch_size=params["batch_size"],
        verbose=True
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}

def config_dict_from_obj(obj):
    """Extract config dict from an object with relevant attributes."""
    return {
        "pred_len": obj.pred_len,
        "epochs": obj.epochs,
        "patience": obj.patience,
        "holding_cost": obj.holding_cost,
        "lead_time": obj.lead_time,
        "ordering_cost": obj.ordering_cost,
        "val_metric": obj.val_metric,
        "seed": int(obj.seed),
        "use_decomposition": bool(getattr(obj, "use_decomposition", False)),
        "decomposition_method": getattr(obj, "decomposition_method", "ma"),
        "seasonal_period": int(getattr(obj, "seasonal_period", 4)),
        "trend_hidden_dim": int(getattr(obj, "trend_hidden_dim", 32)),
        "trend_n_layers": int(getattr(obj, "trend_n_layers", 1)),
        "seasonality_model": getattr(obj, "seasonality_model", getattr(obj, "model_type", "tsmixer")),
        "aggregation_method": getattr(obj, "aggregation_method", "sum"),
        "learnable_aggregation": bool(getattr(obj, "learnable_aggregation", False)),
        "hierarchical_decomposition": bool(getattr(obj, "hierarchical_decomposition", False)),
    }