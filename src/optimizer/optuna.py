import threading
from pathlib import Path

import numpy as np
import optuna
import torch

from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer
from src.data.walk_forward import WalkForwardSplitter
from src.utils.seed import set_seed

# ── Search space ──────────────────────────────────────────────────────────────

SEARCH_SPACE: dict[str, list] = {
    "seq_length": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "n_block":    [1, 2, 3],
    "dropout":    [0.1, 0.3, 0.5, 0.7, 0.9],
    "batch_size": [2, 3, 4],
    "ff_dim":     [8, 16, 32, 64, 128],
    "lr":         [1e-4, 1e-5],
}


class OptunaOptimizer:
    def __init__(
        self,
        scenario: int,
        val_metric_type: str,
        n_trials: int = 50,
        save_dir: str = "checkpoints_optuna",
        pred_len: int = 3,
        epochs: int = 200,
        patience: int = 50,
        holding_cost: float = 2.0,
        lead_time: int = 2,
        forecast_horizon: int = 4,
        ordering_cost: float = 50_000,
        n_jobs: int = 1,
        resume: bool = False,
    ):
        assert scenario in (1, 2), "scenario must be 1 or 2"
        self.scenario        = scenario
        self.n_trials        = n_trials
        self.save_dir        = Path(save_dir)
        self.pred_len        = pred_len
        self.epochs          = epochs
        self.patience        = patience
        self.holding_cost    = holding_cost
        self.lead_time       = lead_time
        self.ordering_cost   = ordering_cost
        self.forecast_horizon = forecast_horizon
        self.val_metric_type = val_metric_type or ("tc" if scenario == 2 else "mape")
        self.n_jobs          = n_jobs
        self.resume          = resume

        self.trial_results: dict = {}
        self._lock      = threading.Lock()
        self.db_path    = f"sqlite:///optuna_s{scenario}.db"
        self.study_name = f"scenario_{scenario}_optimization"

        # ── FIX 5: Cảnh báo nếu dùng SQLite với n_jobs > 1 ───────────────
        if n_jobs > 1:
            print(
                f"⚠️  WARNING: n_jobs={n_jobs} với SQLite có thể gây lỗi 'database is locked'.\n"
                "   Nên dùng PostgreSQL: db_path = 'postgresql://user:pass@localhost/optuna_db'"
            )

    # ── Trainer factory ───────────────────────────────────────────────────────

    def _make_trainer(self, params: dict, trial: optuna.Trial | None = None):
        """Build the appropriate trainer from a params dict, casting types as needed."""
        def safe_int(x, default=1):
            if isinstance(x, (int, float, str)):
                try:
                    return int(float(x))
                except Exception:
                    return default
            return default
        def safe_float(x, default=0.0):
            if isinstance(x, (int, float, str)):
                try:
                    return float(x)
                except Exception:
                    return default
            return default
        def get_int(params, key, default):
            v = params.get(key, default)
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                try:
                    return int(float(v))
                except Exception:
                    return default
            return default
        def get_float(params, key, default):
            v = params.get(key, default)
            if isinstance(v, float):
                return v
            if isinstance(v, int):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    return default
            return default
        seq_length = get_int(params, "seq_length", 1)
        ff_dim = get_int(params, "ff_dim", 8)
        dropout = get_float(params, "dropout", 0.1)
        pred_len = int(self.pred_len)
        n_block = get_int(params, "n_block", 1)
        batch_size = get_int(params, "batch_size", 4)
        lr = get_float(params, "lr", 1e-4)
        epochs = int(self.epochs)
        patience = int(self.patience)
        holding_cost = float(self.holding_cost)
        lead_time = int(self.lead_time)
        ordering_cost = float(self.ordering_cost)
        val_metric_type = str(self.val_metric_type)
        seed = 42
        if self.scenario == 1:
            return Scenario1Trainer(
                seq_length, ff_dim, dropout, pred_len, n_block, batch_size, lr, epochs, patience,
                holding_cost, lead_time, ordering_cost, val_metric_type, seed, trial
            )
        return Scenario2Trainer(
            seq_length, ff_dim, dropout, pred_len, n_block, batch_size, lr, epochs, patience,
            holding_cost, lead_time, ordering_cost, val_metric_type, seed, trial
        )

    # ── Objective ─────────────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        params  = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}
        trainer = self._make_trainer(params, trial=trial)

        walk_params = dict(
            seq_length       = params["seq_length"],
            pred_length      = self.pred_len,
            forecast_horizon = 4,
            train_ratio      = 0.6,
            val_size         = 21,
            test_size        = 21,
            step             = 3,
        )

        mean_val_loss = trainer.crossval_loss_for_optuna(
            walk_params, batch_size=params["batch_size"], verbose=False
        )

        print(f"Trial {trial.number} | val_metric = {mean_val_loss:.4f} | params = {params}")

        with self._lock:
            self.trial_results[trial.number] = {
                "params":     params,
                "val_metric": mean_val_loss,
                "metrics":    {},
            }

        return mean_val_loss

    # ── Save top-3 ────────────────────────────────────────────────────────────

    def _save_top3(self, study: optuna.Study) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        top3 = sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else float("inf"),
        )[:3]

        print(f"\n── Scenario {self.scenario} — Top 3 ─────────────────────────────")
        for rank, trial in enumerate(top3, start=1):
            result  = self.trial_results.get(trial.number, {})
            metrics = result.get("metrics", {})

            torch.save(
                {"rank": rank, "scenario": self.scenario,
                 "val_metric": trial.value, "params": trial.params},
                self.save_dir / f"s{self.scenario}_rank{rank}.pt",
            )

            val_label = "MAPE (%)" if self.val_metric_type == "mape" else "TC_min"
            lines = [
                f"Scenario {self.scenario} — Rank {rank}",
                "=" * 45,
                "",
                "── Loss Configuration ──────────────────────────",
                f"  {'Val Metric':<12}: {self.val_metric_type.upper()}",
                "",
                "── Hyperparameters ─────────────────────────────",
                *[f"  {k:<12}: {v}" for k, v in trial.params.items()],
                "",
                "── Validation metric ────────────────────────────",
                f"  {val_label:<12}: {trial.value:.4f}",
                "",
                "── Test metrics ─────────────────────────────────",
                f"  {'MAE':<12}: {metrics.get('MAE', 0):.4f}",
                f"  {'MSE':<12}: {metrics.get('MSE', 0):.4f}",
                f"  {'RMSE':<12}: {metrics.get('RMSE', 0):.4f}",
                f"  {'MAPE (%)':<12}: {metrics.get('MAPE', 0):.4f}",
                f"  {'TC_min':<12}: {metrics.get('TC_min', 0):.2f}",
                f"  {'c_s*':<12}: {metrics.get('c_s_star', 0):.4f}",
            ]
            (self.save_dir / f"s{self.scenario}_rank{rank}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            print(f"  Rank {rank} | val={trial.value:.4f} | saved → s{self.scenario}_rank{rank}.pt / .txt")

    # ── Callback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _callback(study: optuna.Study, frozen_trial) -> None:
        state = frozen_trial.state
        n     = frozen_trial.number
        if state == optuna.trial.TrialState.COMPLETE:
            print(f"  ✓ Trial {n:3d} | Value: {frozen_trial.value:9.4f} | Best: {study.best_value:9.4f}")
        elif state == optuna.trial.TrialState.PRUNED:
            print(f"  ⊘ Trial {n:3d} | PRUNED")
        elif state == optuna.trial.TrialState.FAIL:
            print(f"  ✗ Trial {n:3d} | FAILED")

        # In tổng hợp mỗi 50 trial
        if (n + 1) % 50 == 0:
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            failed    = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            print("\n===== Optuna Progress Summary =====")
            print(f"Trials finished: {n+1}")
            print(f"  Completed: {completed}")
            print(f"  Pruned   : {pruned}")
            print(f"  Failed   : {failed}")
            print(f"  Best value so far: {study.best_value:.4f}")
            best_trial = study.best_trial
            print(f"  Best params: {best_trial.params}")
            print("===================================\n")

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self) -> optuna.Study:
        set_seed()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=40, n_ei_candidates=48)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=10)

        if self.resume:
            study = optuna.load_study(study_name=self.study_name, storage=self.db_path)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"\n📂 Resumed study: {self.study_name}")
            print(f"   Completed trials : {completed}")
            print(f"   Best value so far: {study.best_value:.4f}")
        else:
            try:
                optuna.delete_study(study_name=self.study_name, storage=self.db_path)
            except KeyError:
                pass
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                storage=self.db_path,
                study_name=self.study_name,
            )
            print(f"\n🆕 Created study : {self.study_name}")
            print(f"   Database       : {self.db_path}")

        print(f"\n⚙️  Configuration:")
        print(f"   Scenario      : {self.scenario}")
        print(f"   Metric        : {self.val_metric_type.upper()}")
        print(f"   Trials        : {self.n_trials}")
        print(f"   Parallel jobs : {self.n_jobs}")
        print(f"\n🔍 Starting optimisation...")

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
            callbacks=[self._callback],
            catch=(RuntimeError, ValueError),
        )

        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed    = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        print(f"\n{'='*60}")
        print(f"✅ Optimisation Complete")
        print(f"   Scenario {self.scenario} | Best val : {study.best_value:.4f}")
        print(f"   Total trials : {len(study.trials)}")
        print(f"   Completed    : {completed}")
        print(f"   Pruned       : {pruned}")
        print(f"   Failed       : {failed}")
        print(f"{'='*60}")

        self._save_top3(study)
        return study


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed()
    OptunaOptimizer(
        scenario=2,
        n_trials=100,
        val_metric_type="tc",
        n_jobs=4,
        resume=True,
    ).run()