from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer


def _decomp_kwargs(params, config, model_type):
    return {
        "use_decomposition": config.get("use_decomposition", False),
        "decomposition_method": params.get("decomposition_method", config.get("decomposition_method", "ma")),
        "seasonal_period": params.get("seasonal_period", config.get("seasonal_period", 4)),
        "trend_hidden_dim": params.get("trend_hidden_dim", config.get("trend_hidden_dim", 32)),
        "trend_n_layers": params.get("trend_n_layers", config.get("trend_n_layers", 1)),
        "seasonality_model": params.get("seasonality_model", config.get("seasonality_model", model_type)),
        "aggregation_method": params.get("aggregation_method", config.get("aggregation_method", "sum")),
        "learnable_aggregation": params.get("learnable_aggregation", config.get("learnable_aggregation", False)),
        "hierarchical_decomposition": params.get("hierarchical_decomposition", config.get("hierarchical_decomposition", False)),
    }


def _seasonality_backbone_kwargs(params):
    return {
        "n_stacks": params.get("seasonality_n_stacks", params.get("n_stacks")),
        "n_blocks": params.get("seasonality_n_blocks", params.get("n_blocks")),
        "n_layers": params.get("seasonality_n_layers", params.get("n_layers")),
        "hidden_dim": params.get("seasonality_hidden_dim", params.get("hidden_dim")),
        "layer_dim": params.get("seasonality_layer_dim", params.get("layer_dim")),
    }

def make_trainer(scenario, params, config, seed=42, trial=None, model_type="tsmixer"):
    """
    Create a trainer instance.
    
    Args:
        scenario: 1 or 2
        params: Hyperparameter dict
        config: Config dict
        seed: Random seed
        trial: Optuna trial object
        model_type: "tsmixer", "nbeats", or "nhits"
    """
    TrainerClass = {
        1: Scenario1Trainer,
        2: Scenario2Trainer
    }[scenario]

    if model_type == "tsmixer":
        seasonality_kwargs = _seasonality_backbone_kwargs(params)
        return TrainerClass(
            seq_length=params["seq_length"],
            ff_dim=params["ff_dim"],
            dropout=params["dropout"],
            pred_len=config["pred_len"],
            n_block=params["n_block"],
            n_stacks=seasonality_kwargs["n_stacks"],
            n_blocks=seasonality_kwargs["n_blocks"],
            n_layers=seasonality_kwargs["n_layers"],
            hidden_dim=seasonality_kwargs["hidden_dim"],
            layer_dim=seasonality_kwargs["layer_dim"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            epochs=config["epochs"],
            patience=config["patience"],
            holding_cost=config["holding_cost"],
            lead_time=config["lead_time"],
            ordering_cost=config["ordering_cost"],
            seed=int(seed),
            trial=trial,
            model_type="tsmixer",
            **_decomp_kwargs(params, config, model_type),
        )
    elif model_type == "nbeats":
        seasonality_kwargs = _seasonality_backbone_kwargs(params)
        return TrainerClass(
            seq_length=params["seq_length"],
            n_stacks=params["n_stacks"],
            n_layers=params["n_layers"],
            layer_dim=params["layer_dim"],
            dropout=params["dropout"],
            n_blocks=seasonality_kwargs["n_blocks"],
            hidden_dim=seasonality_kwargs["hidden_dim"],
            pred_len=config["pred_len"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            epochs=config["epochs"],
            patience=config["patience"],
            holding_cost=config["holding_cost"],
            lead_time=config["lead_time"],
            ordering_cost=config["ordering_cost"],
            seed=int(seed),
            trial=trial,
            model_type="nbeats",
            **_decomp_kwargs(params, config, model_type),
        )
    elif model_type == "nhits":
        seasonality_kwargs = _seasonality_backbone_kwargs(params)
        return TrainerClass(
            seq_length=params["seq_length"],
            n_stacks=params["n_stacks"],
            n_blocks=params["n_blocks"],
            n_layers=params["n_layers"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            layer_dim=seasonality_kwargs["layer_dim"],
            pred_len=config["pred_len"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            epochs=config["epochs"],
            patience=config["patience"],
            holding_cost=config["holding_cost"],
            lead_time=config["lead_time"],
            ordering_cost=config["ordering_cost"],
            seed=int(seed),
            trial=trial,
            model_type="nhits",
            **_decomp_kwargs(params, config, model_type),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")