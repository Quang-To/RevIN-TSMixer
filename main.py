from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer

SCENARIO = 2
VAL_METRIC = "tc"      

SEQ_LENGTH = 12
PRED_LEN = 3
N_BLOCK = 1
FF_DIM = 128
DROPOUT = 0.1

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 2000
PATIENCE = 100

HOLDING_COST = 2
LEAD_TIME = 2
ORDERING_COST = 50_000



def main():
    seeds = [42]  
    for seed in seeds:
        print(f"\n{'='*30} SEED: {seed} {'='*30}")
        TrainerClass = Scenario1Trainer if SCENARIO == 1 else Scenario2Trainer
        trainer = TrainerClass(
            seq_length=SEQ_LENGTH, ff_dim=FF_DIM, dropout=DROPOUT, pred_len=PRED_LEN,
            n_block=N_BLOCK, batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS, patience=PATIENCE,
            holding_cost=HOLDING_COST, lead_time=LEAD_TIME, ordering_cost=ORDERING_COST,
            generate_plots=True,
            val_metric_type=VAL_METRIC,
            seed=seed
        )
        best_val, best_state, metrics = trainer.train()
        trainer.generate_visualizations(metrics)

        print("\n" + "=" * 60)
        print(f"SCENARIO {SCENARIO} - COMPLETE | SEED: {seed}")
        print("=" * 60)
        if SCENARIO == 1:
            print(f"Best Validation MAPE (%): {best_val:.4f}")
        else:
            print(f"Best Validation TC_min: {best_val:.2f}")
        print("-" * 60)
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAPE (%): {metrics['MAPE']:.4f}")
        print(f"TC_min: {metrics['TC_min']:.2f}")
        print(f"c_s*: {metrics['c_s_star']:.4f}")
        print("=" * 60)

        print("\n" + "=" * 80)
        print("TEST PREDICTIONS vs ACTUAL VALUES")
        print("=" * 80)
        print(f"{'Index':<8} {'Predicted':<15} {'Actual':<15} {'Difference':<15} {'% Error':<12}")
        print("-" * 80)
        for i in range(min(30, len(trainer.test_pred))):
            pred = trainer.test_pred[i]
            true = trainer.test_true[i]
            diff = pred - true
            pct_err = abs(diff) / (abs(true) + 1e-8) * 100
            print(f"{i:<8} {pred:<15.2f} {true:<15.2f} {diff:<15.2f} {pct_err:<12.2f}%")
        if len(trainer.test_pred) > 30:
            print(f"\n... ({len(trainer.test_pred) - 30} more values)")
        print("=" * 80)


if __name__ == "__main__":
    main()