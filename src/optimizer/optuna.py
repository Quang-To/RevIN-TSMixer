import torch
import optuna
from pathlib import Path

from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.utils.seed import set_seed

SEARCH_SPACE = {
    "seq_length": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "n_block"   : [1, 2, 3],
    "dropout"   : [0.1, 0.3, 0.5, 0.7, 0.9],
    "batch_size": [1, 2, 3, 4],
    "ff_dim"    : [8, 16, 32, 64, 128],
    "lr"        : [1e-4, 1e-5],
}

class OptunaOptimizer:
    def __init__(self, scenario: int, n_trials: int = 50,
                 save_dir: str = "checkpoints_optuna",
                 pred_len: int = 3, epochs: int = 100, patience: int = 20,
                 holding_cost: int = 2, lead_time: int = 2,
                 ordering_cost: int = 50000):
        assert scenario in (1, 2)
        self.scenario      = scenario
        self.n_trials      = n_trials
        self.save_dir      = Path(save_dir)
        self.pred_len      = pred_len
        self.epochs        = epochs
        self.patience      = patience
        self.holding_cost  = holding_cost
        self.lead_time     = lead_time
        self.ordering_cost = ordering_cost
        self.trial_results = {}  # Store results to avoid re-training

    def _make_trainer(self, params: dict, generate_plots: bool = False):
        shared = dict(
            seq_length=params["seq_length"], ff_dim=params["ff_dim"],
            dropout=params["dropout"],       pred_len=self.pred_len,
            n_block=params["n_block"],       batch_size=params["batch_size"],
            lr=params["lr"],                 epochs=self.epochs,
            patience=self.patience,          holding_cost=self.holding_cost,
            lead_time=self.lead_time,        ordering_cost=self.ordering_cost,
            generate_plots=generate_plots,
        )
        return Scenario1Trainer(**shared) if self.scenario == 1 else Scenario2Trainer(**shared)

    def _objective(self, trial: optuna.Trial) -> float:
        set_seed(trial.number)
        params  = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}
        trainer = self._make_trainer(params)
        best_val, best_state, metrics = trainer.train()
        
        # Auto-save results to avoid re-training
        self.trial_results[trial.number] = {
            "params": params,
            "val_metric": best_val,
            "model_state": best_state,
            "metrics": metrics,
        }
        return best_val

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
            val_label = "MAPE (%)" if self.scenario == 1 else "TC_min"
            lines = [
                f"Scenario {self.scenario} — Rank {rank}",
                "=" * 45,
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

    def run(self) -> optuna.Study:
        set_seed()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self._objective, n_trials=self.n_trials,
                       show_progress_bar=True, catch=(Exception,))

        print(f"\n{'='*50}")
        print(f"Scenario {self.scenario} | Best val: {study.best_value:.4f}")
        print(f"{'='*50}")
        self._save_top3(study)
        return study

if __name__ == "__main__":
    set_seed()
    OptunaOptimizer(scenario=2, n_trials=50).run()