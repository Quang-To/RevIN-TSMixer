from src.hpo.config import DEFAULT_WALK_PARAMS, SEARCH_SPACE
from src.hpo.trainer_factory import make_trainer

def sample_params(trial):
    """Sample hyperparameters from Optuna trial."""
    return {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}

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
        "seed": int(obj.seed)
    }