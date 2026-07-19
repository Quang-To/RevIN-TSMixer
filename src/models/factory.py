# src/models/factory.py

import config

def resolve_model_kwargs(best_params: dict, decomp_params: dict, ckpt_decomp_params: dict, model_kwargs: dict) -> dict:
    """Override model hyperparameters with decomposition equivalents if present in checkpoint or best_params."""
    for key, base_key in config.KEY_MAPPING.items():
        if base_key in model_kwargs:
            if key in ckpt_decomp_params or key in best_params or base_key in best_params:
                model_kwargs[base_key] = decomp_params[key]
    return model_kwargs


def build_trainer(
    model_name: str,
    best_params: dict,
    decomp_params: dict,
    TrainerClass,
    visualizer,
    ckpt_decomp_params: dict = None,
):
    if ckpt_decomp_params is None:
        ckpt_decomp_params = {}

    common = dict(
        pred_len     = best_params.get("pred_len", best_params.get("pred_length", config.PRED_LEN)),
        batch_size   = best_params.get("batch_size", config.BATCH_SIZE),
        lr           = best_params.get("lr", config.LR),
        epochs       = best_params.get("epochs", config.EPOCHS),
        patience     = best_params.get("patience", config.PATIENCE),
        holding_cost = best_params.get("holding_cost", config.HOLDING_COST),
        ordering_cost= best_params.get("ordering_cost", config.ORDERING_COST),
        lead_time    = best_params.get("lead_time", config.LEAD_TIME),
        seed         = best_params.get("seed", config.SEED),
        visualizer   = visualizer,
        use_log_return = best_params.get("use_log_return", config.USE_LOG_RETURN),
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

    if config.FORCE_SUM_AGGREGATION and decomp["use_decomposition"]:
        decomp["aggregation_method"] = "sum"
        decomp["learnable_aggregation"] = False

    if model_name == "tsmixer":
        model_kwargs = dict(
            seq_length = best_params.get("seq_length", config.SEQ_LENGTH),
            ff_dim     = best_params.get("ff_dim", config.FF_DIM),
            n_block    = best_params.get("n_block", config.N_BLOCK),
            dropout    = best_params.get("dropout", config.DROPOUT),
            model_type = "tsmixer",
        )
        model_kwargs = resolve_model_kwargs(best_params, decomp_params, ckpt_decomp_params, model_kwargs)
        all_kwargs = {**common, **decomp, **model_kwargs}
        return TrainerClass(**all_kwargs)

    elif model_name == "nbeats":
        model_kwargs = dict(
            seq_length = best_params.get("seq_length", config.SEQ_LENGTH),
            n_stacks   = best_params.get("n_stacks", config.N_STACKS),
            n_layers   = best_params.get("n_layers", config.N_LAYERS),
            layer_dim  = best_params.get("layer_dim", config.LAYER_DIM),
            dropout    = best_params.get("dropout", config.DROPOUT),
            model_type = "nbeats",
        )
        model_kwargs = resolve_model_kwargs(best_params, decomp_params, ckpt_decomp_params, model_kwargs)
        all_kwargs = {**common, **decomp, **model_kwargs}
        return TrainerClass(**all_kwargs)

    elif model_name == "nhits":
        model_kwargs = dict(
            seq_length = best_params.get("seq_length", config.SEQ_LENGTH),
            n_stacks   = best_params.get("n_stacks", config.N_STACKS),
            n_blocks   = best_params.get("n_blocks", config.N_BLOCKS),
            n_layers   = best_params.get("n_layers", config.N_LAYERS),
            hidden_dim = best_params.get("hidden_dim", config.HIDDEN_DIM),
            dropout    = best_params.get("dropout", config.DROPOUT),
            model_type = "nhits",
        )
        model_kwargs = resolve_model_kwargs(best_params, decomp_params, ckpt_decomp_params, model_kwargs)
        all_kwargs = {**common, **decomp, **model_kwargs}
        return TrainerClass(**all_kwargs)

    else:
        raise ValueError(f"Unknown MODEL: {model_name!r}. Must be 'tsmixer', 'nbeats', or 'nhits'")
