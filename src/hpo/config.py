DEFAULT_SPLIT_PARAMS = {
    "train_size":       84,
    "val_size":         12,
    "test_size":        24,
    "forecast_horizon": 1,
}
DEFAULT_WALK_PARAMS = DEFAULT_SPLIT_PARAMS

VALID_METRICS           = {"mape", "pinball", "tc", "mae", "rmse"}
VALID_MODELS            = {"tsmixer", "nbeats", "nhits"}
DEFAULT_N_TRIALS        = 100
DEFAULT_N_STARTUP_TRIALS = 10
DEFAULT_N_EI_CANDIDATES  = 24
SUMMARY_EVERY_N          = 20
