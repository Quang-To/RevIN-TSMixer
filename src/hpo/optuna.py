import sys
from pathlib import Path

# Add project root to sys.path to support direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import threading
import optuna
import torch
import logging
import numpy as np
from src.utils.seed import set_seed
from src.hpo.config import VALID_METRICS, VALID_MODELS, DEFAULT_N_STARTUP_TRIALS, DEFAULT_N_EI_CANDIDATES, SUMMARY_EVERY_N
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
        model_type: str = "tsmixer",
        use_decomposition: bool = False,
        decomposition_method: str = "stl",
        seasonal_period: int = 4,
        stl_robust: bool = True,
        stl_seasonal: int = 7,
        stl_trend: int | None = None,
        stl_low_pass: int | None = None,
        trend_hidden_dim: int = 32,
        trend_n_layers: int = 1,
        seasonality_model: str | None = None,
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
        use_log_return: bool = False,
    ):
        assert scenario in (1, 2), "scenario must be 1 or 2"
        assert model_type in VALID_MODELS, f"model_type must be one of {VALID_MODELS}, got '{model_type}'"
        self.scenario        = scenario
        self.model_type      = model_type
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
        self.use_decomposition = bool(use_decomposition)
        self.decomposition_method = decomposition_method
        self.seasonal_period = int(seasonal_period)
        self.stl_robust = bool(stl_robust)
        self.stl_seasonal = int(stl_seasonal)
        self.stl_trend = stl_trend
        self.stl_low_pass = stl_low_pass
        self.trend_hidden_dim = int(trend_hidden_dim)
        self.trend_n_layers = int(trend_n_layers)
        self.seasonality_model = seasonality_model or model_type
        self.aggregation_method = aggregation_method
        self.learnable_aggregation = bool(learnable_aggregation)
        self.hierarchical_decomposition = bool(hierarchical_decomposition)
        self.use_log_return = bool(use_log_return)

        self.trial_results: dict = {}
        self._lock      = threading.Lock()
        mode_tag = "decomp" if self.use_decomposition else "base"
        if self.use_log_return:
            mode_tag = f"{mode_tag}_logret"
        self.db_path    = f"sqlite:///optuna_s{scenario}_{model_type}_{mode_tag}.db"
        self.study_name = f"scenario_{scenario}_{model_type}_{mode_tag}_optimization"

    def _log_trial(self, trial, params, loss):
        logger.debug(f"Trial {trial.number} | val_metric = {loss:.4f} | params = {params}")

    def _config_dict(self):
        return config_dict_from_obj(self)

    def _trial_checkpoint_path(self, trial_number: int) -> Path:
        return self.save_dir / f"s{self.scenario}_{self.model_type}_trial{trial_number}.pt"

    def _normalize_fold_best_epochs(self, fold_best_epochs):
        if not fold_best_epochs:
            return []
        return [int(epoch) for epoch in fold_best_epochs if epoch is not None]

    def _build_checkpoint_payload(self, trial, params, loss, metrics, *, rank: int | None = None, final_result=None):
        fold_best_epochs = self._normalize_fold_best_epochs(
            metrics.get("fold_best_epochs", []) if isinstance(metrics, dict) else []
        )
        fold_losses = metrics.get("fold_losses", []) if isinstance(metrics, dict) else []
        median_best_epoch = int(round(float(np.median(fold_best_epochs)))) if fold_best_epochs else None

        payload = {
            "trial_number": getattr(trial, "number", None),
            "value": float(loss),
            "val_metric": float(loss),
            "params": params,
            "metrics": metrics,
            "fold_best_epochs": fold_best_epochs,
            "best_epoch": median_best_epoch,
            "cv_median_best_epoch": median_best_epoch,
            "fold_losses": fold_losses,
            "cv_mean": float(metrics.get("mean", float(np.mean(fold_losses)) if fold_losses else float("inf"))) if isinstance(metrics, dict) else float("inf"),
            "cv_std": float(metrics.get("std", float(np.std(fold_losses)) if fold_losses else 0.0)) if isinstance(metrics, dict) else 0.0,
            "cv_min": float(metrics.get("min", float(np.min(fold_losses)) if fold_losses else float("inf"))) if isinstance(metrics, dict) else float("inf"),
            "cv_max": float(metrics.get("max", float(np.max(fold_losses)) if fold_losses else float("inf"))) if isinstance(metrics, dict) else float("inf"),
        }
        if rank is not None:
            payload["rank"] = rank
        if final_result is not None:
            payload["final_result"] = final_result
        return payload

    def _checkpoint_matches_trial(self, payload, trial) -> bool:
        if not isinstance(payload, dict):
            return False
        payload_trial_number = payload.get("trial_number")
        if payload_trial_number is not None:
            return payload_trial_number == trial.number
        return payload.get("params") == getattr(trial, "params", {})

    def _save_trial_checkpoint(self, trial, params, loss, metrics) -> None:
        try:
            payload = self._build_checkpoint_payload(trial, params, loss, metrics)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self._trial_checkpoint_path(trial.number))
        except Exception:
            logger.exception(f"Failed to persist trial checkpoint for trial {trial.number}")

    def _load_trial_checkpoint(self, trial):
        candidates = [self._trial_checkpoint_path(trial.number)]
        candidates.extend(sorted(self.save_dir.glob(f"s{self.scenario}_{self.model_type}_rank*.pt")))

        for path in candidates:
            if not path.exists():
                continue
            try:
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(path, map_location="cpu")
            except Exception:
                continue
            if self._checkpoint_matches_trial(payload, trial):
                return payload
        return None

    def _build_trial_summary(self, trial):
        result = self.trial_results.get(trial.number, {})
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        fold_best_epochs = self._normalize_fold_best_epochs(metrics.get("fold_best_epochs") if isinstance(metrics, dict) else [])
        fallback_value = float(trial.value) if trial.value is not None else float("inf")

        checkpoint = None
        if not metrics or not fold_best_epochs:
            checkpoint = self._load_trial_checkpoint(trial)

        if checkpoint and not metrics:
            metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint.get("metrics"), dict) else {}
            fold_best_epochs = self._normalize_fold_best_epochs(checkpoint.get("fold_best_epochs"))
            if not fold_best_epochs and checkpoint.get("best_epoch") is not None:
                fold_best_epochs = [int(checkpoint.get("best_epoch"))]

        if not metrics:
            ua = trial.user_attrs if hasattr(trial, "user_attrs") else {}
            fold_losses = ua.get("fold_losses") if isinstance(ua, dict) else []
            if not fold_losses and checkpoint:
                fold_losses = checkpoint.get("fold_losses", [])
            fold_best_epochs = self._normalize_fold_best_epochs(
                (ua.get("fold_best_epochs") if isinstance(ua, dict) else None)
                or (checkpoint.get("fold_best_epochs") if checkpoint else None)
            )
            if not fold_best_epochs and checkpoint and checkpoint.get("best_epoch") is not None:
                fold_best_epochs = [int(checkpoint.get("best_epoch"))]
            metrics = {
                "mean": (ua.get("cv_mean") if isinstance(ua, dict) else None)
                if isinstance(ua, dict) and ua.get("cv_mean") is not None
                else (checkpoint.get("cv_mean") if checkpoint and checkpoint.get("cv_mean") is not None else (float(np.mean(fold_losses)) if fold_losses else fallback_value)),
                "std": (ua.get("cv_std") if isinstance(ua, dict) else None)
                if isinstance(ua, dict) and ua.get("cv_std") is not None
                else (checkpoint.get("cv_std") if checkpoint and checkpoint.get("cv_std") is not None else (float(np.std(fold_losses)) if fold_losses else 0.0)),
                "min": (ua.get("cv_min") if isinstance(ua, dict) else None)
                if isinstance(ua, dict) and ua.get("cv_min") is not None
                else (checkpoint.get("cv_min") if checkpoint and checkpoint.get("cv_min") is not None else (float(np.min(fold_losses)) if fold_losses else fallback_value)),
                "max": (ua.get("cv_max") if isinstance(ua, dict) else None)
                if isinstance(ua, dict) and ua.get("cv_max") is not None
                else (checkpoint.get("cv_max") if checkpoint and checkpoint.get("cv_max") is not None else (float(np.max(fold_losses)) if fold_losses else fallback_value)),
                "fold_losses": fold_losses,
                "fold_best_epochs": fold_best_epochs,
            }

        if not fold_best_epochs:
            fold_best_epochs = self._normalize_fold_best_epochs(
                (trial.user_attrs.get("fold_best_epochs") if hasattr(trial, "user_attrs") else None)
                or (checkpoint.get("fold_best_epochs") if checkpoint else None)
            )
            if not fold_best_epochs and checkpoint and checkpoint.get("best_epoch") is not None:
                fold_best_epochs = [int(checkpoint.get("best_epoch"))]

        return metrics, fold_best_epochs, checkpoint

    def _objective(self, trial):
        torch.set_num_threads(1)
        params = sample_params(trial, model_type=self.model_type, use_decomposition=self.use_decomposition)
        
        # Assign GPU dynamically if multiple are available
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_id = trial.number % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
            
        trainer = make_trainer(
            self.scenario,
            params,
            self._config_dict(),
            seed=self.seed,
            trial=trial,
            model_type=self.model_type,
            device=device
        )
        walk_params = build_walk_params(params, self.pred_len)
        loss, metrics = evaluate(trainer, walk_params, params)
        # Persist fold-level epoch info into Optuna trial user attributes so it
        # survives process restarts (resume).
        try:
            fold_best_epochs = metrics.get("fold_best_epochs", []) if isinstance(metrics, dict) else []
            trial.set_user_attr("fold_best_epochs", fold_best_epochs)
            trial.set_user_attr("cv_median_best_epoch", int(round(float(np.median(fold_best_epochs)))) if fold_best_epochs else None)
            if isinstance(metrics, dict):
                trial.set_user_attr("fold_losses", metrics.get("fold_losses", []))
                trial.set_user_attr("cv_mean", float(metrics.get("mean", float("inf"))))
                trial.set_user_attr("cv_std", float(metrics.get("std", 0.0)))
                trial.set_user_attr("cv_min", float(metrics.get("min", float("inf"))))
                trial.set_user_attr("cv_max", float(metrics.get("max", float("inf"))))
        except Exception:
            # non-critical if cannot set user attrs (e.g., trial is not a real Optuna trial)
            pass

        with self._lock:
            self.trial_results[trial.number] = {"metrics": metrics}
        self._save_trial_checkpoint(trial, params, loss, metrics)
        self._log_trial(trial, params, loss)
        return loss

    # ── Save top-3 ────────────────────────────────────────────────────────────

    def _save_top3(self, study: optuna.Study) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        top3 = sorted(
            completed_trials,
            key=lambda t: t.value if t.value is not None else float("inf"),
        )[:3]

        if not top3:
            logger.info(f"\n── Scenario {self.scenario} — Model {self.model_type.upper()} — No COMPLETE trials to save.")
            return

        logger.info(f"\n── Scenario {self.scenario} — Model {self.model_type.upper()} — Top 3 ──────────────────────────")
        for rank, trial in enumerate(top3, start=1):
            metrics, fold_best_epochs, checkpoint = self._build_trial_summary(trial)

            median_best_epoch = int(round(float(np.median(fold_best_epochs)))) if fold_best_epochs else None
            final_train_epochs = median_best_epoch
            # Run final training with the fold-wise best epochs to get final test metrics
            final_result = None
            try:
                trainer = make_trainer(self.scenario, trial.params, self._config_dict(), seed=self.seed, trial=None, model_type=self.model_type)
                walk_params = build_walk_params(trial.params, self.pred_len)
                final_result = trainer.train_and_test_with_best_hparams(
                    walk_params,
                    batch_size=trial.params.get("batch_size"),
                    verbose=False,
                    fold_best_epochs=fold_best_epochs,
                    fixed_epochs=final_train_epochs,
                )
            except Exception as e:
                # don't fail the whole saving process if final training fails
                logger.exception(f"Final training failed for trial {trial.number}: {e}")

            payload = self._build_checkpoint_payload(trial, trial.params, trial.value, metrics, rank=rank, final_result=final_result)
            payload.update({
                "scenario": self.scenario,
                "model_type": self.model_type,
                "final_train_epochs": final_train_epochs,
            })
            torch.save(payload, self.save_dir / f"s{self.scenario}_{self.model_type}_rank{rank}.pt")

            val_label = "MAPE (%)" if self.val_metric == "mape" else "TC_min"
            lines = [
                f"Scenario {self.scenario} — Model {self.model_type.upper()} — Rank {rank}",
                "=" * 50,
                "",
                "── Loss Configuration ──────────────────────────",
                f"  {'Val Metric':<12}: {self.val_metric.upper()}",
                f"  {'Model':<12}: {self.model_type.upper()}",
                "",
                "── Hyperparameters ─────────────────────────────",
                *[f"  {k:<12}: {v}" for k, v in trial.params.items()],
                "",
                "── Fold best epochs ────────────────────────────",
                f"  {'epochs':<12}: {fold_best_epochs}",
                f"  {'median':<12}: {median_best_epoch}",
                f"  {'final epochs':<12}: {final_train_epochs}",
                "",
                "── Validation metric (CV mean) ──────────────────",
                f"  {val_label:<12}: {trial.value:.4f}",
                "",
                "── CV fold statistics ───────────────────────────",
                f"  {'fold mean':<12}: {metrics.get('mean', 0):.4f}",
                f"  {'fold std':<12}: {metrics.get('std', 0):.4f}",
                f"  {'fold min':<12}: {metrics.get('min', 0):.4f}",
                f"  {'fold max':<12}: {metrics.get('max', 0):.4f}",
            ]

            # Append final evaluation summary if available
            if final_result is not None:
                final_metrics = final_result.get('metrics', {})
                lines.extend([
                    "",
                    "── Final evaluation ───────────────────────────",
                    f"  {'final_epochs':<12}: {final_result.get('best_epoch', 'N/A')}",
                    f"  {'final_Tc_min':<12}: {final_metrics.get('TC_min', 0):.4f}",
                    f"  {'final_MAPE':<12}: {final_metrics.get('MAPE', 0):.4f}",
                ])
            (self.save_dir / f"s{self.scenario}_{self.model_type}_rank{rank}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            logger.info(f"  Rank {rank} | val={trial.value:.4f} | saved → s{self.scenario}_{self.model_type}_rank{rank}.pt / .txt")

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
        torch.set_num_threads(1)
        set_seed()
        optuna.logging.set_verbosity(optuna.logging.INFO)

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=DEFAULT_N_STARTUP_TRIALS, n_ei_candidates=DEFAULT_N_EI_CANDIDATES)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=10)

        if self.resume:
            try:
                study = optuna.load_study(study_name=self.study_name, storage=self.db_path)
                completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                logger.info(f"\n📂 Resumed study: {self.study_name}")
                logger.info(f"   Completed trials : {completed}")
                if completed > 0:
                    logger.info(f"   Best value so far: {study.best_value:.4f}")
            except KeyError:
                logger.info(f"\nℹ️ Study '{self.study_name}' not found. Creating a new study instead.")
                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                    storage=self.db_path,
                    study_name=self.study_name,
                )
                logger.info(f"\n🆕 Created study : {self.study_name}")
                logger.info(f"   Database       : {self.db_path}")
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
        logger.info(f"   Model         : {self.model_type.upper()}")
        logger.info(f"   Decomposition : {self.use_decomposition}")
        logger.info(f"   Log-return    : {self.use_log_return}")
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

        best_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        best_val_text = f"{study.best_value:.4f}" if best_trials else "N/A"

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Optimisation Complete")
        logger.info(f"   Scenario {self.scenario} | Best val : {best_val_text}")
        logger.info(f"   Total trials : {len(study.trials)}")
        logger.info(f"   Completed    : {completed}")
        logger.info(f"   Pruned       : {pruned}")
        logger.info(f"   Failed       : {failed}")
        logger.info(f"{'='*60}")

        self._save_top3(study)
        return study

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization.")
    parser.add_argument("--model", type=str, default="tsmixer", choices=["tsmixer", "nbeats", "nhits"], help="Model type to optimize.")
    parser.add_argument("--scenario", type=int, default=2, choices=[1, 2], help="Scenario number (1 or 2).")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("--use_decomposition", action="store_true", help="Enable decomposition branch (STL).")
    parser.add_argument("--use_log_return", action="store_true", help="Enable log-return transformation on target Quantity.")
    parser.add_argument("--resume", action="store_true", help="Resume previous study if exists.")
    
    args = parser.parse_args()
    
    val_metric = "mape" if args.scenario == 1 else "tc"
    
    OptunaOptimizer(
        scenario=args.scenario,
        n_trials=args.n_trials,
        val_metric=val_metric,
        n_jobs=args.n_jobs,
        resume=args.resume,
        model_type=args.model,
        use_decomposition=args.use_decomposition,
        use_log_return=args.use_log_return,
    ).run()