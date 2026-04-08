import torch
import optuna
from pathlib import Path
from datetime import datetime

from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.seed import set_seed

# OPTIMIZED search space (narrowed based on paper results)
SEARCH_SPACE = {
    "seq_length": [7, 8, 9],              # ← narrowed (was 1-9) | paper: 9
    "n_block"   : [1, 2, 3],              # keep all
    "dropout"   : [0.3, 0.5, 0.7],       # ← narrowed (was 0.1-0.9) | paper: 0.5
    "batch_size": [1, 2, 3, 4],           # keep all
    "ff_dim"    : [32, 64, 128],         # ← narrowed (was 8-128) | paper: 64
    "lr"        : [1e-4, 1e-5],           # keep all
}
# Combinations: 3×3×3×4×3×2 = 216 (was 5,400 = 0.9% coverage → now 23% with 50 trials)

class OptunaOptimizer:
    def __init__(self, scenario: int, val_metric_type: str, 
                 n_trials: int = 50,
                 save_dir: str = "checkpoints_optuna",
                 pred_len: int = 3, epochs: int = 200, patience: int = 50,
                 holding_cost: int = 2, lead_time: int = 2,
                 ordering_cost: int = 50000, n_jobs: int = 1,
                 resume: bool = False):
        assert scenario in (1, 2)
        self.scenario       = scenario
        self.n_trials       = n_trials
        self.save_dir       = Path(save_dir)
        self.pred_len       = pred_len
        self.epochs         = epochs
        self.patience       = patience
        self.holding_cost   = holding_cost
        self.lead_time      = lead_time
        self.ordering_cost  = ordering_cost
        self.val_metric_type = val_metric_type or ("tc" if scenario == 2 else "mape")
        self.n_jobs         = n_jobs
        self.resume         = resume
        self.trial_results  = {}
        self.db_path        = f"sqlite:///optuna_s{scenario}.db"
        self.study_name     = f"scenario_{scenario}_optimization"  

    def _make_trainer(self, params: dict, trial_number: int, trial: optuna.Trial = None, generate_plots: bool = False):
        shared = dict(
            seq_length=params["seq_length"], ff_dim=params["ff_dim"],
            dropout=params["dropout"],       pred_len=self.pred_len,
            n_block=params["n_block"],       batch_size=params["batch_size"],
            lr=params["lr"],                 epochs=self.epochs,
            patience=self.patience,          holding_cost=self.holding_cost,
            lead_time=self.lead_time,        ordering_cost=self.ordering_cost,
            generate_plots=generate_plots,
            val_metric_type=self.val_metric_type,
            seed=42,                         # ← FIXED seed (was trial_number) for reproducible results
            trial=trial,
        )
        return Scenario1Trainer(**shared) if self.scenario == 1 else Scenario2Trainer(**shared)

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            params  = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}
            trainer = self._make_trainer(params, trial_number=trial.number, trial=trial)
            best_val, best_state, metrics = trainer.train()
            
            # Auto-save results to avoid re-training
            self.trial_results[trial.number] = {
                "params": params,
                "val_metric": best_val,
                "model_state": best_state,
                "metrics": metrics,
            }
            return best_val
        except optuna.exceptions.TrialPruned:
            raise  # Re-raise pruned trials

    def _save_top3(self, study: optuna.Study) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        top3 = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))[:3]

        print(f"\n── Scenario {self.scenario} — Top 3 ─────────────────────────────")
        for rank, trial in enumerate(top3, start=1):
            # Load stored results instead of re-training
            result = self.trial_results.get(trial.number, {})
            best_state = result.get("model_state")
            metrics = result.get("metrics", {})

            print(f"  Rank {rank} | val={trial.value:.4f} | saved")

            # ── checkpoint .pt ────────────────────────────────────────────────
            torch.save({
                "rank":        rank,
                "scenario":    self.scenario,
                "val_metric":  trial.value,
                "params":      trial.params,
                "model_state": best_state,
            }, self.save_dir / f"s{self.scenario}_rank{rank}.pt")

            # ── report .txt ───────────────────────────────────────────────────
            txt_path = self.save_dir / f"s{self.scenario}_rank{rank}.txt"
            val_label = "MAPE (%)" if self.val_metric_type == "mape" else "TC_min"
            lines = [
                f"Scenario {self.scenario} — Rank {rank}",
                "=" * 45,
                "",
                "── Loss Configuration ──────────────────────────",
                f"  {'Val Metric':<12}: {self.val_metric_type.upper()}",
                "",
                "── Hyperparameters ──────────────────────────",
                *[f"  {k:<12}: {v}" for k, v in trial.params.items()],
                "",
                "── Validation metric ────────────────────────",
                f"  {val_label:<12}: {trial.value:.4f}",
                "",
                "── Test metrics ─────────────────────────────",
                f"  {'MAE':<12}: {metrics.get('MAE', 0):.4f}",
                f"  {'MSE':<12}: {metrics.get('MSE', 0):.4f}",
                f"  {'RMSE':<12}: {metrics.get('RMSE', 0):.4f}",
                f"  {'MAPE (%)':<12}: {metrics.get('MAPE', 0):.4f}",
                f"  {'TC_min':<12}: {metrics.get('TC_min', 0):.2f}",
                f"  {'c_s*':<12}: {metrics.get('c_s_star', 0):.4f}",
            ]
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Saved → s{self.scenario}_rank{rank}.pt / .txt")

    def _callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback for runtime monitoring"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"  ✓ Trial {trial.number:3d} | Value: {trial.value:9.4f} | Best: {study.best_value:9.4f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"  ⊘ Trial {trial.number:3d} | PRUNED")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"  ✗ Trial {trial.number:3d} | FAILED")

    def run(self) -> optuna.Study:
        set_seed()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # ── 1. IMPROVED TPE SAMPLER ─────────────────────────────────────────
        sampler = optuna.samplers.TPESampler(
            seed=42,                      # Fixed seed for reproducibility
            n_startup_trials=40,          # ← INCREASED (was 10) = 40% exploration
            n_ei_candidates=48,           # Better EI quality
        )
        
        # ── 2. PRUNER (Early Stopping) ───────────────────────────────────
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,  # Number of trials before pruning kicks in
        )
        
        # ── 3. DATABASE PERSISTENCE ──────────────────────────────────────
        if self.resume:
            # Load existing study
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.db_path
            )
            print(f"\n📂 Resumed study: {self.study_name}")
            print(f"   Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
            print(f"   Best value so far: {study.best_value:.4f}")
        else:
            # Create new study
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                storage=self.db_path,
                study_name=self.study_name,
                load_if_exists=False,  # Delete if exists
            )
            print(f"\n🆕 Created study: {self.study_name}")
            print(f"   Database: {self.db_path}")
        
        print(f"\n⚙️  Configuration:")
        print(f"   Scenario: {self.scenario}")
        print(f"   Metric: {self.val_metric_type.upper()}")
        print(f"   Trials: {self.n_trials}")
        print(f"   Parallel jobs: {self.n_jobs}")
        print(f"\n🔍 Starting optimization ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})...\n")
        
        # ── 4. PARALLEL EXECUTION + CALLBACKS ────────────────────────────
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,  # Parallel execution
            show_progress_bar=True,
            callbacks=[self._callback],  # Runtime monitoring
            catch=(Exception,),  # Don't stop on exceptions
        )

        print(f"\n{'='*60}")
        print(f"✅ Optimization Complete")
        print(f"   Scenario {self.scenario} | Best val: {study.best_value:.4f}")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"   Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"   Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"{'='*60}")
        self._save_top3(study)
        return study

if __name__ == "__main__":
    set_seed()
    OptunaOptimizer(
        scenario=2,
        n_trials=100,         
        val_metric_type="tc",
        n_jobs=4,           
        resume=False,       
    ).run()