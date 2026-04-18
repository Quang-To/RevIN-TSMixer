from src.data.walk_forward import WalkForwardSplitter

if __name__ == "__main__":
    seq_length = 12
    pred_length = 3
    forecast_horizon = 4
    train_ratio = 0.6
    val_size = 21
    test_size = 21
    step = 3

    splitter = WalkForwardSplitter(
        seq_length=seq_length,
        pred_length=pred_length,
        forecast_horizon=forecast_horizon,
        train_ratio=train_ratio,
        val_size=val_size,
        test_size=test_size,
        step=step
    )

    splitter.summary()
    print(f"\nTổng số folds: {splitter.n_splits()}")