# config.py

SCENARIO   = 2
MODEL      = "nbeats"   # "tsmixer", "nbeats", or "nhits"
VAL_METRIC = "tc"      # "mape" for Scenario 1, "tc" for Scenario 2
SEED       = 42

# Model hyperparameters
SEQ_LENGTH = 11
PRED_LEN   = 3

# TSMixer specific
N_BLOCK    = 2
FF_DIM     = 64

# NBEATS specific
N_STACKS   = 3
N_LAYERS   = 4
LAYER_DIM  = 256

# NHITS specific
N_BLOCKS   = 2
HIDDEN_DIM = 256

# Common
DROPOUT    = 0.1

# Training
BATCH_SIZE = 3
LR         = 0.001
EPOCHS     = 3000
PATIENCE   = 100

# Inventory cost
HOLDING_COST     = 2.0
LEAD_TIME        = 2
ORDERING_COST    = 50_000.0
FORECAST_HORIZON = 4

# RevIN
USE_LOG_RETURN = False

# Decomposition defaults
FORCE_NO_DECOMPOSITION = False
FORCE_SUM_AGGREGATION = False

DECOMP_DEFAULTS = dict(
    decomposition_method      = "stl",
    seasonal_period           = 4,
    stl_robust                = True,
    stl_seasonal              = 7,
    stl_trend                 = None,
    stl_low_pass              = None,
    trend_hidden_dim          = 32,
    trend_n_layers            = 2,           
    seasonality_n_stacks      = 2,
    seasonality_n_blocks      = 1,
    seasonality_n_layers      = 4,
    seasonality_hidden_dim    = 256,
    seasonality_layer_dim     = 256,
    seasonality_ff_dim        = 64,
    seasonality_n_block       = 2,
    aggregation_method        = "weighted",
    learnable_aggregation     = False,
    hierarchical_decomposition= False,
)

KEY_MAPPING = {
    "seasonality_n_stacks": "n_stacks",
    "seasonality_n_blocks": "n_blocks",
    "seasonality_n_layers": "n_layers",
    "seasonality_hidden_dim": "hidden_dim",
    "seasonality_layer_dim": "layer_dim",
    "seasonality_ff_dim": "ff_dim",
    "seasonality_n_block": "n_block",
}
