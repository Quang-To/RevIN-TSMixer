from src.trainers import Scenario1Trainer, Scenario2Trainer


def make_trainer(scenario: int, params: dict, config: dict,
                 seed: int = 42, trial=None,
                 model_type: str = "tsmixer", device=None):

    TrainerClass = {1: Scenario1Trainer, 2: Scenario2Trainer}[scenario]

    lr = params.get("lr", params.get("learning_rate", 1e-3))
    seq_length = params.get("seq_length", params.get("lookback", 24))
    hidden = params.get("hidden_size", params.get("ff_dim", params.get("layer_dim", params.get("hidden_dim", 256))))
    n_layers = params.get("n_layers", params.get("n_block", 3))
    dropout = params.get("dropout", 0.1)
    revin_affine = params.get("revin_affine", True)

    common = dict(
        pred_len      = config.get("pred_len", 1),
        batch_size    = params.get("batch_size", 16),
        lr            = lr,
        epochs        = config.get("epochs", 300),
        patience      = config.get("patience", 30),
        holding_cost  = config.get("holding_cost", 4.0),
        shortage_cost = config.get("shortage_cost", 30.0),
        lead_time     = config.get("lead_time", 1),
        ordering_cost = config.get("ordering_cost", 50000.0),
        seed          = int(seed),
        trial         = trial,
        model_type    = model_type,
        use_log_return= config.get("use_log_return", False),
        data_file     = config.get("data_file", None),
        revin_affine  = revin_affine,
        device        = device,
    )

    if model_type == "tsmixer":
        return TrainerClass(
            seq_length = seq_length,
            ff_dim     = hidden,
            n_block    = n_layers,
            dropout    = dropout,
            **common,
        )

    elif model_type == "nbeats":
        return TrainerClass(
            seq_length = seq_length,
            n_stacks   = params.get("n_stacks", 2),
            n_blocks   = params.get("n_blocks", 2),
            n_layers   = n_layers,
            layer_dim  = hidden,
            dropout    = dropout,
            **common,
        )

    elif model_type == "nhits":
        return TrainerClass(
            seq_length = seq_length,
            n_stacks   = params.get("n_stacks", 2),
            n_blocks   = params.get("n_blocks", 2),
            n_layers   = n_layers,
            hidden_dim = hidden,
            dropout    = dropout,
            **common,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
