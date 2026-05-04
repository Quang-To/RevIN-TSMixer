from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer

def make_trainer(scenario, params, config, seed=42, trial=None, model_type="tsmixer"):
    """
    Create a trainer instance.
    
    Args:
        scenario: 1 or 2
        params: Hyperparameter dict
        config: Config dict
        seed: Random seed
        trial: Optuna trial object
        model_type: "tsmixer" or "nbeats"
    """
    TrainerClass = {
        1: Scenario1Trainer,
        2: Scenario2Trainer
    }[scenario]

    if model_type == "tsmixer":
        return TrainerClass(
            seq_length=params["seq_length"],
            ff_dim=params["ff_dim"],
            dropout=params["dropout"],
            pred_len=config["pred_len"],
            n_block=params["n_block"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            epochs=config["epochs"],
            patience=config["patience"],
            holding_cost=config["holding_cost"],
            lead_time=config["lead_time"],
            ordering_cost=config["ordering_cost"],
            seed=int(seed),
            trial=trial,
            model_type="tsmixer"
        )
    elif model_type == "nbeats":
        return TrainerClass(
            seq_length=params["seq_length"],
            n_stacks=params["n_stacks"],
            n_layers=params["n_layers"],
            layer_dim=params["layer_dim"],
            dropout=params["dropout"],
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
            model_type="nbeats"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")