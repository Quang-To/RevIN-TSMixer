# TSMixer Search Space
SEARCH_SPACE_TSMIXER: dict[str, list] = {
    "seq_length": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "n_block":    [1, 2, 3],
    "dropout":    [0.1, 0.3, 0.5, 0.7, 0.9],
    "batch_size": [2, 3, 4],
    "ff_dim":     [8, 16, 32, 64, 128],
    "lr":         [1e-4, 1e-5],
}

# NBEATS Search Space
SEARCH_SPACE_NBEATS: dict[str, list] = {
    "seq_length": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "n_stacks":   [2, 3, 4],
    "n_layers":   [2, 3, 4],
    "layer_dim":  [64, 128, 256],
    "dropout":    [0.1, 0.3, 0.5],
    "batch_size": [2, 3, 4],
    "lr":         [1e-4, 1e-5],
}

# NHITS Search Space
SEARCH_SPACE_NHITS: dict[str, list] = {
    "seq_length": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    "dropout":    [0.1, 0.3, 0.5, 0.7, 0.9],
    "batch_size": [2, 3, 4],
    "lr":         [1e-4, 1e-5],
    "n_stacks":   [2, 3],
    "n_blocks":   [1, 2, 3],
    "n_layers":   [2, 3, 4],
    "hidden_dim": [64, 128, 256],
}

SEARCH_SPACE: dict[str, list] = SEARCH_SPACE_TSMIXER

DEFAULT_WALK_PARAMS = {
    "train_ratio": 0.6,
    "val_size": 21,
    "test_size": 21,
    "step": 3
}

VALID_METRICS = {"mape", "tc", "mae", "rmse"}
VALID_MODELS = {"tsmixer", "nbeats", "nhits"}
DEFAULT_N_STARTUP_TRIALS = 40
DEFAULT_N_EI_CANDIDATES = 48
SUMMARY_EVERY_N = 50