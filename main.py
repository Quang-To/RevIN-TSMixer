from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer

# ── Configuration ─────────────────────────────────────────────────────────────
# Select scenario:
#   1 → optimise MAPE
#   2 → optimise inventory total cost (TC_min)
SCENARIO = 2

# Model hyperparameters
SEQ_LENGTH   = 9        # lookback window (number of past months)
N_BLOCK      = 2        # number of MixerBlocks
DROPOUT      = 0.5      # dropout rate
FF_DIM       = 64       # hidden dimension of FeatureMixingLayer
PRED_LEN     = 3        # forecast horizon (months ahead)

# Training hyperparameters
BATCH_SIZE   = 4
LR           = 1e-4     # learning rate
EPOCHS       = 1000
PATIENCE     = 100      # early-stopping patience

# Inventory cost parameters
HOLDING_COST  = 2
LEAD_TIME     = 2
ORDERING_COST = 50_000
# ──────────────────────────────────────────────────────────────────────────────


def main():
    TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer
    trainer = TrainerClass(
        seq_length=SEQ_LENGTH,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        pred_len=PRED_LEN,
        n_block=N_BLOCK,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        patience=PATIENCE,
        holding_cost=HOLDING_COST,
        lead_time=LEAD_TIME,
        ordering_cost=ORDERING_COST,
    )
    best_val, best_state, metrics = trainer.train()

    val_label = "MAPE (%)" if SCENARIO == 1 else "TC_min"
    print("\n" + "=" * 50)
    print(f"  SCENARIO {SCENARIO} — TRAINING COMPLETE")
    print("=" * 50)
    print(f"  {'Best val ' + val_label:<20}: {best_val:>15.4f}")
    print("-" * 50)
    print("  TEST METRICS")
    print("-" * 50)
    print(f"  {'MAE':<20}: {metrics['MAE']:>15.4f}")
    print(f"  {'MSE':<20}: {metrics['MSE']:>15.4f}")
    print(f"  {'RMSE':<20}: {metrics['RMSE']:>15.4f}")
    print(f"  {'MAPE (%)':<20}: {metrics['MAPE']:>14.4f} %")
    print(f"  {'TC_min':<20}: {metrics['TC_min']:>15.2f}")
    print(f"  {'c_s*':<20}: {metrics['c_s_star']:>15.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()