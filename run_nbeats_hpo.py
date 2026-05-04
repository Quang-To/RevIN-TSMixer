#!/usr/bin/env python
"""
Run Optuna hyperparameter optimization for NBEATS model.
"""

from src.hpo.optuna import OptunaOptimizer

if __name__ == "__main__":
    # ── Configuration ──────────────────────────────────────────────────────
    
    optimizer = OptunaOptimizer(
        scenario=2,                    # Scenario 1 or 2
        val_metric="tc",               # "tc" for Scenario 2, "mape" for Scenario 1
        n_trials=100,                  # Number of trials
        model_type="nbeats",           # ← NBEATS model
        n_jobs=4,                      # Parallel jobs (set to 1 if GPU issues)
        seed=42,
        
        # Model & Training params
        pred_len=3,
        epochs=300,                    # Max epochs per fold
        patience=50,                   # Early stopping patience
        
        # Inventory costs
        holding_cost=2.0,
        lead_time=2,
        ordering_cost=50_000.0,
        
        # Database & checkpoints
        save_dir="checkpoints_optuna",
        resume=False,                  # Set True to resume previous HPO
    )
    
    # ── Run Optimization ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🚀 Starting NBEATS Hyperparameter Optimization")
    print("="*60)
    
    study = optimizer.run()
    
    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅ Optimization Complete!")
    print("="*60)
    
    best_trial = study.best_trial
    print(f"\n📊 Best Trial: {best_trial.number}")
    print(f"   Value (metric): {best_trial.value:.4f}")
    print(f"\n   Best NBEATS Parameters:")
    for key, value in best_trial.params.items():
        print(f"     - {key}: {value}")
    
    print(f"\n💾 Checkpoints saved to: checkpoints_optuna/")
    print(f"   - s2_nbeats_rank1.pt  (best)")
    print(f"   - s2_nbeats_rank2.pt  (2nd)")
    print(f"   - s2_nbeats_rank3.pt  (3rd)")
    print(f"\n📁 Database: optuna_s2_nbeats.db")
