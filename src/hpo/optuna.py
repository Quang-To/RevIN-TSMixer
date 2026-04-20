import threading
from pathlib import Path
import optuna
import torch
import logging
from src.utils.seed import set_seed
from src.hpo.config import VALID_METRICS, DEFAULT_N_STARTUP_TRIALS, DEFAULT_N_EI_CANDIDATES, SUMMARY_EVERY_N
from src.hpo.trainer_factory import make_trainer
from src.hpo.objective import sample_params, build_walk_params, evaluate, config_dict_from_obj

logger = logging.getLogger("optuna_optimizer")
logging.basicConfig(level=logging.INFO)

class OptunaOptimizer:
    def __init__(
        self,
        scenario: int,
        val_metric: str,
        n_trials: int = 50,
        save_dir: str = "checkpoints_optuna",
        pred_len: int = 3,
        epochs: int = 3000,
        patience: int = 100,
        holding_cost: float = 2.0,
        lead_time: int = 2,
        forecast_horizon: int = 4,
        ordering_cost: float = 50_000,
        n_jobs: int = 1,
        resume: bool = False,
        seed: int = 42,
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
        if val_metric and val_metric not in VALID_METRICS:
            raise ValueError(f"val_metric must be one of {VALID_METRICS}, got '{val_metric}'")
        self.val_metric = val_metric or ("tc" if scenario == 2 else "mape")
        self.n_jobs          = n_jobs
        self.resume          = resume
        self.seed            = int(seed)

        self.trial_results: dict = {}
        self._lock      = threading.Lock()
        self.db_path    = f"sqlite:///optuna_s{scenario}.db"
        self.study_name = f"scenario_{scenario}_optimization"

    def _log_trial(self, trial, params, loss):
        print(f"Trial {trial.number} | val_metric = {loss:.4f} | params = {params}")

    def _config_dict(self):
        return config_dict_from_obj(self)

    def _objective(self, trial):
        params = sample_params(trial)
        trainer = make_trainer(self.scenario, params, self._config_dict(), seed=self.seed, trial=trial)
        walk_params = build_walk_params(params, self.pred_len)
        loss, metrics = evaluate(trainer, walk_params, params)

        # Lưu avg_best_epoch vào trial để dùng cho final train sau này
        avg_best_epoch = getattr(trainer, "_cv_avg_best_epoch", self.epochs)
        trial.set_user_attr("avg_best_epoch", avg_best_epoch)
        trial.set_user_attr("fold_best_epochs", getattr(trainer, "_cv_best_epochs", []))

        with self._lock:
            self.trial_results[trial.number] = {"metrics": metrics}
        self._log_trial(trial, params, loss)
        return loss

    # ── Save top-3 ────────────────────────────────────────────────────────────

    def _save_top3(self, study: optuna.Study) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        top3 = sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else float("inf"),
        )[:3]

        print(f"\n── Scenario {self.scenario} — Top 3 ─────────────────────────────")
        for rank, trial in enumerate(top3, start=1):
            result           = self.trial_results.get(trial.number, {})
            metrics          = result.get("metrics", {})
            best_epoch       = trial.user_attrs.get("avg_best_epoch", self.epochs)
            fold_best_epochs = trial.user_attrs.get("fold_best_epochs", [])

            torch.save(
                {
                    "rank":             rank,
                    "scenario":         self.scenario,
                    "val_metric":       trial.value,
                    "params":           trial.params,
                    "best_epoch":       best_epoch,
                    "fold_best_epochs": fold_best_epochs,
                },
                self.save_dir / f"s{self.scenario}_rank{rank}.pt",
            )

            val_label = "MAPE (%)" if self.val_metric == "mape" else "TC_min"
            lines = [
                f"Scenario {self.scenario} — Rank {rank}",
                "=" * 45,
                "",
                "── Loss Configuration ──────────────────────────",
                f"  {'Val Metric':<12}: {self.val_metric.upper()}",
                "",
                "── Hyperparameters ─────────────────────────────",
                *[f"  {k:<12}: {v}" for k, v in trial.params.items()],
                "",
                "── Validation metric (CV mean) ──────────────────",
                f"  {val_label:<12}: {trial.value:.4f}",
                f"  {'Best epoch':<12}: {best_epoch}",
                f"  {'Per-fold':<12}: {fold_best_epochs}",
                "",
                "── CV fold statistics ───────────────────────────",
                f"  {'fold mean':<12}: {metrics.get('mean', 0):.4f}",
                f"  {'fold std':<12}: {metrics.get('std', 0):.4f}",
                f"  {'fold min':<12}: {metrics.get('min', 0):.4f}",
                f"  {'fold max':<12}: {metrics.get('max', 0):.4f}",
            ]
            (self.save_dir / f"s{self.scenario}_rank{rank}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            print(f"  Rank {rank} | val={trial.value:.4f} | saved → s{self.scenario}_rank{rank}.pt / .txt")

    # ── Callback ──────────────────────────────────────────────────────────────

    def _make_callback(self):
        def callback(study, frozen_trial):
            state = frozen_trial.state
            n     = frozen_trial.number
            if state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"  ✓ Trial {n:3d} | Value: {frozen_trial.value:9.4f} | Best: {study.best_value:9.4f}")
            elif state == optuna.trial.TrialState.PRUNED:
                logger.info(f"  ⊘ Trial {n:3d} | PRUNED")
            elif state == optuna.trial.TrialState.FAIL:
                logger.info(f"  ✗ Trial {n:3d} | FAILED")

            if (n + 1) % SUMMARY_EVERY_N == 0:
                completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                failed    = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
                logger.info("\n===== Optuna Progress Summary =====")
                logger.info(f"Trials finished: {n+1}")
                logger.info(f"  Completed: {completed}")
                logger.info(f"  Pruned   : {pruned}")
                logger.info(f"  Failed   : {failed}")
                logger.info(f"  Best value so far: {study.best_value:.4f}")
                best_trial = study.best_trial
                logger.info(f"  Best params: {best_trial.params}")
                logger.info("===================================\n")
        return callback

    def run(self) -> optuna.Study:
        set_seed()
        optuna.logging.set_verbosity(optuna.logging.INFO)

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=DEFAULT_N_STARTUP_TRIALS, n_ei_candidates=DEFAULT_N_EI_CANDIDATES)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=10)

        if self.resume:
            study = optuna.load_study(study_name=self.study_name, storage=self.db_path)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            logger.info(f"\n📂 Resumed study: {self.study_name}")
            logger.info(f"   Completed trials : {completed}")
            if completed > 0:
                logger.info(f"   Best value so far: {study.best_value:.4f}")
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
            logger.info(f"\n🆕 Created study : {self.study_name}")
            logger.info(f"   Database       : {self.db_path}")

        logger.info(f"\n⚙️  Configuration:")
        logger.info(f"   Scenario      : {self.scenario}")
        logger.info(f"   Metric        : {self.val_metric.upper()}")
        logger.info(f"   Trials        : {self.n_trials}")
        logger.info(f"   Parallel jobs : {self.n_jobs}")
        logger.info(f"\n🔍 Starting optimisation...")

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
            callbacks=[self._make_callback()],
            catch=(RuntimeError, ValueError),
        )

        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed    = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Optimisation Complete")
        logger.info(f"   Scenario {self.scenario} | Best val : {study.best_value:.4f}")
        logger.info(f"   Total trials : {len(study.trials)}")
        logger.info(f"   Completed    : {completed}")
        logger.info(f"   Pruned       : {pruned}")
        logger.info(f"   Failed       : {failed}")
        logger.info(f"{'='*60}")

        self._save_top3(study)
        return study

if __name__ == "__main__":
    OptunaOptimizer(
        scenario=2,
        n_trials=100,
        val_metric="tc",
        n_jobs=4,
        resume=False,
    ).run()