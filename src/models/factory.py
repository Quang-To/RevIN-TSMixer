import torch.nn as nn
import config
from src.models.ForecastModel.NBEATSModel.NBEATSModel import NBEATSModel
from src.models.ForecastModel.NHITSModel.NHITSModel import NHITSModel
from src.models.ForecastModel.TSMixer.ForecastModel import ForecastModel
from src.models.ForecastModel.PredictionHead.QuantileHead import DEFAULT_QUANTILES

MULTIVARIATE_MODELS = ["tsmixer", "dlinear"]


def build_model(name: str, lookback: int, horizon: int = 1, n_quantiles: int = len(DEFAULT_QUANTILES),
                hidden_size: int = 256, n_layers: int = 3, dropout: float = 0.1,
                revin_affine: bool = True, **kw) -> nn.Module:
    name = name.lower()
    if name == "nbeats":
        return NBEATSModel(
            seq_length=lookback, pred_len=horizon,
            n_stacks=kw.get("n_stacks", 2), n_blocks=kw.get("n_blocks", 2),
            layer_dim=hidden_size, n_layers=n_layers, dropout=dropout,
            revin_affine=revin_affine,
        )
    elif name == "nhits":
        return NHITSModel(
            seq_length=lookback, pred_len=horizon,
            hidden_dim=hidden_size, n_layers=n_layers, dropout=dropout,
            pool_kernels=kw.get("pool_kernels", (1, 2, 4)),
            revin_affine=revin_affine,
        )
    elif name == "tsmixer":
        return ForecastModel(
            seq_length=lookback, ff_dim=hidden_size, dropout=dropout,
            pred_len=horizon, n_block=n_layers, n_features=kw.get("n_features", 7),
            revin_affine=revin_affine,
        )
    else:
        raise ValueError(f"Mô hình không hỗ trợ: {name}")


def build_model_multi(name: str, lookback: int, n_features: int, horizon: int = 1,
                      n_quantiles: int = len(DEFAULT_QUANTILES), hidden_size: int = 256,
                      n_layers: int = 3, dropout: float = 0.1, revin_affine: bool = False,
                      **kw) -> nn.Module:
    name = name.lower()
    if name == "tsmixer":
        return ForecastModel(
            seq_length=lookback, ff_dim=hidden_size, dropout=dropout,
            pred_len=horizon, n_block=n_layers, n_features=n_features,
            revin_affine=revin_affine,
        )
    else:
        raise ValueError(f"Mô hình đa biến không hỗ trợ: {name}")


def build_trainer(model_name: str, best_params: dict, TrainerClass, visualizer=None):
    common = dict(
        pred_len      = best_params.get("pred_len", best_params.get("pred_length", config.PRED_LEN)),
        batch_size    = best_params.get("batch_size", config.BATCH_SIZE),
        lr            = best_params.get("lr", best_params.get("learning_rate", config.LR)),
        epochs        = best_params.get("epochs", config.EPOCHS),
        patience      = best_params.get("patience", config.PATIENCE),
        holding_cost  = best_params.get("holding_cost", config.HOLDING_COST),
        shortage_cost = best_params.get("shortage_cost", config.SHORTAGE_COST),
        ordering_cost = best_params.get("ordering_cost", config.ORDERING_COST),
        lead_time     = best_params.get("lead_time", config.LEAD_TIME),
        seed          = best_params.get("seed", config.SEED),
        use_log_return= best_params.get("use_log_return", config.USE_LOG_RETURN),
        revin_affine  = best_params.get("revin_affine", True),
        data_file     = best_params.get("data_file", getattr(config, "DATA_FILE", None)),
        visualizer    = visualizer,
    )

    seq_len = best_params.get("seq_length", best_params.get("lookback", config.SEQ_LENGTH))
    dropout = best_params.get("dropout", config.DROPOUT)
    hidden = best_params.get("hidden_size", best_params.get("ff_dim", best_params.get("hidden_dim", best_params.get("layer_dim", 256))))
    n_layers = best_params.get("n_layers", best_params.get("n_block", 3))

    if model_name.lower() == "tsmixer":
        return TrainerClass(
            seq_length = seq_len,
            ff_dim     = hidden,
            n_block    = n_layers,
            dropout    = dropout,
            model_type = "tsmixer",
            **common,
        )

    elif model_name.lower() == "nbeats":
        return TrainerClass(
            seq_length = seq_len,
            n_stacks   = best_params.get("n_stacks", config.N_STACKS),
            n_blocks   = best_params.get("n_blocks", best_params.get("nbeats_n_blocks", config.NBEATS_N_BLOCKS)),
            n_layers   = n_layers,
            layer_dim  = hidden,
            dropout    = dropout,
            model_type = "nbeats",
            **common,
        )

    elif model_name.lower() == "nhits":
        return TrainerClass(
            seq_length = seq_len,
            n_stacks   = best_params.get("n_stacks", config.N_STACKS),
            n_blocks   = best_params.get("n_blocks", config.N_BLOCKS),
            n_layers   = n_layers,
            hidden_dim = hidden,
            dropout    = dropout,
            model_type = "nhits",
            **common,
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}. Chọn: 'tsmixer', 'nbeats', 'nhits'")
