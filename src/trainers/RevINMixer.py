import numpy as np
import torch
from torch.optim import Adam
from typing import Optional
from abc import ABC, abstractmethod
import optuna as _optuna
import logging

from src.utils.seed import set_seed
from src.utils.metrics import mape_loss, compute_metrics, sweep_tc, collect_predictions
from src.models.ForecastModel.ForecastModel import ForecastModel
from src.models.NBEATSModel.NBEATSModel import NBEATSModel
from src.models.NHITSModel.NHITSModel import NHITSModel
from src.models.Decomposition.DecomposedForecastModel import DecomposedForecastModel, HierarchicalDecomposedModel
from src.data.dataset import TimeSeriesData
from src.data.walk_forward import WalkForwardSplitter
from src.data.preprocessing import Preprocessing
from src.utils.decomposition_helpers import (
    custom_stl_decompose,
    custom_decomposed_forecast_forward,
    custom_decomposed_forecast_get_component_forecasts,
    decompose_stl,
    extend_decomposition,
    get_or_compute_decomposition
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_FORECAST_HORIZON = 4
GRAD_CLIP_NORM = 5.0
PRED_CLIP_MIN = 1.0
DIVERGE_THRESHOLD = 2.0  # Increased from 1.5 to prevent premature pruning


# ── Base trainer ──────────────────────────────────────────────────────────────

class BaseTrainer(ABC):
    def __init__(self, seq_length: int, batch_size: int, lr: float, epochs: int, patience: int,
        holding_cost: float, lead_time: int, ordering_cost: float, pred_len: int,
        scenario: int = 1, val_metric_type: str = "mape", seed: int = 42, trial: Optional[object] = None,
        visualizer=None, model_type: str = "tsmixer",
        # TSMixer parameters
        ff_dim: Optional[int] = None,
        n_block: Optional[int] = None,
        dropout: Optional[float] = None,
        # NBEATS parameters
        n_stacks: Optional[int] = None,
        n_layers: Optional[int] = None,
        layer_dim: Optional[int] = None,
        # NHITS parameters
        n_blocks: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        # Decomposed model parameters
        use_decomposition: bool = False,
        decomposition_method: str = "stl",
        seasonal_period: int = 4,
        stl_robust: bool = True,
        stl_seasonal: int = 7,
        stl_trend: Optional[int] = None,
        stl_low_pass: Optional[int] = None,
        trend_hidden_dim: Optional[int] = None,
        trend_n_layers: int = 1,
        seasonality_model: str = "tsmixer",
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
        use_log_return: bool = False,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.holding_cost = holding_cost
        self.lead_time = lead_time
        self.ordering_cost = ordering_cost
        self.pred_len = pred_len
        self.scenario = scenario
        self.val_metric_type = val_metric_type
        self.seed = int(seed)
        self.trial = trial
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = visualizer
        self.model_type = model_type

        # TSMixer parameters
        self.ff_dim = ff_dim or 64
        self.n_block = n_block or 1
        self.dropout = dropout or 0.1

        # NBEATS parameters
        self.n_stacks = n_stacks or 3
        self.n_layers = n_layers or 4
        self.layer_dim = layer_dim or 128

        # NHITS parameters
        self.n_blocks = n_blocks or 1
        self.hidden_dim = hidden_dim or 64
        
        # Decomposed model parameters
        self.use_decomposition = use_decomposition
        self.decomposition_method = decomposition_method
        self.seasonal_period = seasonal_period
        self.stl_robust = bool(stl_robust)
        self.stl_seasonal = int(stl_seasonal)
        self.stl_trend = stl_trend
        self.stl_low_pass = stl_low_pass
        self.trend_hidden_dim = trend_hidden_dim or 32
        self.trend_n_layers = trend_n_layers
        self.seasonality_model = seasonality_model
        self.aggregation_method = aggregation_method
        self.learnable_aggregation = learnable_aggregation
        self.hierarchical_decomposition = hierarchical_decomposition
        self.use_log_return = use_log_return
        
        # Pre-decomposed arrays for leak-free forecasting
        self.y_full = Preprocessing().preprocess()["Quantity"].values
        self.trend_total = None
        self.seasonal_total = None
        self.resid_total = None
    OPTUNA_PATIENCE = 50

    def crossval_loss_for_optuna(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        min_epochs: int = 20,
    ) -> tuple[float, dict]:
        # Use forecast_horizon from walk_params when present. Fall back to
        # pred_length (as provided by build_walk_params) or the default.
        fh = walk_params.get("forecast_horizon")
        if fh is None:
            fh = walk_params.get("pred_length", DEFAULT_FORECAST_HORIZON)
        walk_params = {**walk_params, "forecast_horizon": fh}
        bsz = batch_size or self.batch_size
        splitter = WalkForwardSplitter(**walk_params)

        fold_losses = []
        fold_best_epochs = []

        orig_epochs, orig_patience = self.epochs, self.patience
        self.patience = self.OPTUNA_PATIENCE

        try:
            for fold_idx, split in enumerate(splitter.get_splits()):
                train_loader = self._make_loader(0, split["train_end"], bsz)
                val_loader   = self._make_loader(split["train_end"], split["val_end"], bsz)
                set_seed(self.seed)
                model     = self._build_model()
                optimizer = Adam(model.parameters(), lr=self.lr)

                best_val, best_epoch = self._train_one_fold(
                    model, optimizer, train_loader, val_loader,
                    fold_idx=fold_idx,
                    max_epochs=None,
                    min_epochs=min_epochs,
                    verbose=verbose,
                )

                fold_losses.append(best_val)
                fold_best_epochs.append(best_epoch)

                if verbose:
                    logging.debug(f"    [Fold {fold_idx}] best_val={best_val:.4f}")
        finally:
            self.epochs   = orig_epochs
            self.patience = orig_patience

        mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        metrics = {
            "mean":       mean_loss,
            "std":        float(np.std(fold_losses))  if fold_losses else 0.0,
            "min":        float(np.min(fold_losses))  if fold_losses else float("inf"),
            "max":        float(np.max(fold_losses))  if fold_losses else float("inf"),
            "fold_losses": fold_losses,
            "fold_best_epochs": fold_best_epochs,
        }
        return mean_loss, metrics

    def train_and_test_with_best_hparams(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        n_epochs: int = 3000,
        patience: int = 100,
        fold_best_epochs: Optional[list[int]] = None,
        fixed_epochs: Optional[int] = None,
    ) -> dict:
        """
        Final training run after Optuna selects best hyperparameters.
        Trains on ALL data before the test window (train + val combined)
        for a fixed number of epochs derived from the fold-wise best epochs,
        then evaluates exactly once on the held-out test window.
        Evaluates once on the held-out test window.
        """
        bsz              = batch_size or self.batch_size
        forecast_horizon = walk_params.get("forecast_horizon", DEFAULT_FORECAST_HORIZON)
        splitter         = WalkForwardSplitter(**walk_params)
        final            = splitter.get_final_test()
        if fixed_epochs is None:
            if fold_best_epochs:
                fixed_epochs = int(round(float(np.median(fold_best_epochs))))
            else:
                fixed_epochs = int(n_epochs)
        fixed_epochs = max(1, int(fixed_epochs))

        train_loader = self._make_loader(0, final["train_end"], bsz, forecast_horizon)
        test_loader  = self._make_loader(final["test_start"], final["test_end"], bsz, forecast_horizon)

        if verbose:
            logging.debug(f"  [Train] [0:{final['train_end']}] — fixed epochs={fixed_epochs}")

        set_seed(self.seed)
        model     = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        for epoch in range(1, fixed_epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)

            if self.visualizer is not None:
                self.visualizer.log_epoch(epoch, train_loss, None)

            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == fixed_epochs):
                logging.info(f"  [Train] Epoch {epoch:>4}/{fixed_epochs} | train={train_loss:.4f}")

        train_errors                       = self._collect_train_errors(model, train_loader)
        test_pred, test_true, test_indices = self._run_inference(model, test_loader)
        test_metrics                       = self._compute_results(test_pred, test_true, train_errors)
        decomp_components                  = self._collect_decomp_components(model, test_loader) if self.use_decomposition else None

        if verbose:
            logging.debug(f"\n[Result] Epochs={fixed_epochs} | TC_min={test_metrics['TC_min']:.4f} | MAPE={test_metrics['MAPE']:.4f}%")

        return {
            "metrics":      test_metrics,
            "test_pred":    test_pred,
            "test_true":    test_true,
            "test_indices": test_indices,
            "best_epoch":   fixed_epochs,
            "fold_best_epochs": fold_best_epochs or [],
            "decomp_components": decomp_components,
        }

    def train_walk_forward(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: int = 1,
    ) -> dict:
        """
        Full walk-forward training pipeline:
          Phase 1 — Cross-validate to find avg_best_epoch.
          Phase 2 — Retrain on all pre-test data and evaluate once on test.
        """
        walk_params = {**walk_params, "forecast_horizon": DEFAULT_FORECAST_HORIZON}
        bsz         = batch_size or self.batch_size
        splitter    = WalkForwardSplitter(**walk_params)

        # ── Phase 1: cross-validation ─────────────────────────────────────────
        val_folds       = []
        best_global_val = float("inf")

        for split in splitter.get_splits():
            fold = split["fold"]
            if verbose >= 1:
                logging.debug(f"\n[Fold {fold}] train_end={split['train_end']} | val_end={split['val_end']}")

            train_loader = self._make_loader(0, split["train_end"], bsz)
            val_loader   = self._make_loader(split["train_end"], split["val_end"], bsz)

            set_seed(self.seed)
            model     = self._build_model()
            optimizer = Adam(model.parameters(), lr=self.lr)

            best_val, best_epoch = self._train_one_fold(
                model, optimizer, train_loader, val_loader,
                fold_idx=fold,
                max_epochs=self.epochs,
                verbose=(verbose >= 2),
            )

            val_folds.append({"fold": fold, "best_val": best_val, "best_epoch": best_epoch})
            best_global_val = min(best_global_val, best_val)

            if verbose >= 1:
                logging.debug(f"  → Best val: {best_val:.4f} (epoch {best_epoch})")

        avg_best_epoch = int(round(np.median([f["best_epoch"] for f in val_folds]))) if val_folds else self.epochs
        if verbose >= 1 and val_folds:
            mean_val = np.mean([f["best_val"] for f in val_folds])
            logging.debug(f"\n[CV Summary] mean val={mean_val:.4f} | avg_best_epoch={avg_best_epoch}")

        # ── Phase 2: retrain on all pre-test data ─────────────────────────────
        final        = splitter.get_final_test()
        train_loader = self._make_loader(0, final["train_end"], bsz)
        test_loader  = self._make_loader(final["test_start"], final["test_end"], bsz)

        if verbose >= 1:
            logging.debug(f"\n[Phase2] Retrain [0:{final['train_end']}] — {avg_best_epoch} epochs")

        set_seed(self.seed)
        model     = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        for epoch in range(1, avg_best_epoch + 1):
            self._train_epoch(model, optimizer, train_loader)

        train_errors                       = self._collect_train_errors(model, train_loader)
        test_pred, test_true, test_indices = self._run_inference(model, test_loader)
        test_metrics                       = self._compute_results(test_pred, test_true, train_errors)

        if verbose >= 1:
            logging.debug(f"\n[Test] TC_min={test_metrics['TC_min']:.4f} | MAPE={test_metrics['MAPE']:.4f}%")

        return {
            "val_folds": val_folds,
            "test": {
                "metrics":      test_metrics,
                "test_pred":    test_pred,
                "test_true":    test_true,
                "test_indices": test_indices,
                "best_val":     best_global_val,
                "best_epoch":   avg_best_epoch,
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train_one_fold(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        fold_idx: int = 0,
        max_epochs: Optional[int] = None,
        min_epochs: int = 1,
        verbose: bool = False,
    ) -> tuple[float, int]:
        """
        Train for up to `self.epochs` with early stopping on val_metric.

        Returns:
            best_val:   Best validation metric seen.
            best_epoch: Epoch at which best_val was achieved.

        Raises:
            optuna.exceptions.TrialPruned: If divergence detected or Optuna requests pruning.
        """
        best_val, best_epoch, no_improve = float("inf"), 1, 0

        epoch = 0
        while True:
            epoch += 1
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_metric = self._val_metric(model, val_loader)

            if verbose and (epoch == 1 or epoch % 100 == 0 or (max_epochs is not None and epoch == max_epochs)):
                total_epochs = max_epochs if max_epochs is not None else "∞"
                logging.debug(f"    [Fold {fold_idx}] Epoch {epoch:>4}/{total_epochs} | train={train_loss:.4f} | val={val_metric:.4f}")

            if val_metric < best_val:
                best_val, best_epoch, no_improve = val_metric, epoch, 0
            else:
                no_improve += 1

            if epoch >= min_epochs and val_metric > best_val * DIVERGE_THRESHOLD:
                if verbose:
                    logging.debug(f"    [Fold {fold_idx}] Diverging → prune")
                raise _optuna.exceptions.TrialPruned()

            if no_improve >= self.patience:
                if verbose:
                    logging.debug(f"    [Fold {fold_idx}] Early stop epoch {epoch}, best_epoch={best_epoch}")
                break

            if max_epochs is not None and epoch >= max_epochs:
                if verbose:
                    logging.debug(f"    [Fold {fold_idx}] Reached max_epochs={max_epochs}, best_epoch={best_epoch}")
                break

            self._maybe_report_to_optuna(fold_idx, epoch, val_metric)

        return best_val, best_epoch

    def _maybe_report_to_optuna(self, fold_idx: int, epoch: int, val_metric: float) -> None:
        """Report intermediate value to Optuna and prune if requested."""
        if self.trial is not None and isinstance(self.trial, _optuna.trial.Trial):
            global_step = fold_idx * self.epochs + epoch
            self.trial.report(val_metric, step=global_step)
            if self.trial.should_prune():
                raise _optuna.exceptions.TrialPruned()

    def _make_loader(
        self,
        split_start: int,
        split_end: int,
        batch_size: Optional[int] = None,
        forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    ):
        if self.use_decomposition and self.decomposition_method == "stl" and split_start == 0:
            t_end = split_end
            period = self.seasonal_period if self.seasonal_period is not None else 4
            seasonal = self.stl_seasonal if self.stl_seasonal is not None else 7
            trend = self.stl_trend if self.stl_trend is not None else 41
            
            # Fetch pre-computed or cached decomposition
            self.trend_total, self.seasonal_total, self.resid_total = get_or_compute_decomposition(
                self.y_full, t_end, period, seasonal, trend
            )

        data = TimeSeriesData(
            seq_length=self.seq_length,
            batch_size=batch_size or self.batch_size,
            pred_length=self.pred_len,
            split_start=split_start,
            split_end=split_end,
            forecast_horizon=forecast_horizon,
            use_log_return=self.use_log_return,
        )
        return data.get_loader()

    def _build_model(self):
        model = self._build_model_raw()
        if self.use_decomposition and self.decomposition_method == "stl" and hasattr(model, "decomposition"):
            import types
            model.decomposition.y_full = self.y_full
            model.decomposition.trend_total = self.trend_total
            model.decomposition.seasonal_total = self.seasonal_total
            model.decomposition.resid_total = self.resid_total
            model.decomposition.forecast_horizon = getattr(self, "forecast_horizon", 4)
            
            # Bind the custom methods
            model.decomposition._stl_decompose = types.MethodType(custom_stl_decompose, model.decomposition)
            model.forward = types.MethodType(custom_decomposed_forecast_forward, model)
            model.get_component_forecasts = types.MethodType(custom_decomposed_forecast_get_component_forecasts, model)
        return model

    def _build_model_raw(self):
        # Decomposed models
        if self.use_decomposition:
            if self.hierarchical_decomposition:
                return HierarchicalDecomposedModel(
                    seq_length=self.seq_length,
                    pred_len=self.pred_len,
                    n_features=1,
                    seasonal_period=self.seasonal_period,
                    dropout=self.dropout,
                    seasonality_model=self.seasonality_model
                ).to(self.device)
            else:
                return DecomposedForecastModel(
                    seq_length=self.seq_length,
                    pred_len=self.pred_len,
                    n_features=1,
                    seasonal_period=self.seasonal_period,
                    decomposition_method=self.decomposition_method,
                    stl_robust=self.stl_robust,
                    stl_seasonal=self.stl_seasonal,
                    stl_trend=self.stl_trend,
                    stl_low_pass=self.stl_low_pass,
                    trend_hidden_dim=self.trend_hidden_dim,
                    trend_n_layers=self.trend_n_layers,
                    seasonality_model=self.seasonality_model,
                    seasonality_hidden_dim=self.hidden_dim,
                    seasonality_n_blocks=self.n_blocks,
                    seasonality_n_stacks=self.n_stacks,
                    seasonality_n_layers=self.n_layers,
                    seasonality_layer_dim=self.layer_dim,
                    seasonality_ff_dim=self.ff_dim,
                    seasonality_n_block=self.n_block,
                    dropout=self.dropout,
                    aggregation_method=self.aggregation_method,
                    learnable_aggregation=self.learnable_aggregation
                ).to(self.device)
        
        # Standard models
        if self.model_type == "tsmixer":
            return ForecastModel(
                self.seq_length, self.ff_dim, self.dropout, self.pred_len, self.n_block
            ).to(self.device)
        elif self.model_type == "nbeats":
            return NBEATSModel(
                self.seq_length, self.pred_len,
                n_stacks=self.n_stacks,
                n_layers=self.n_layers,
                layer_dim=self.layer_dim,
                dropout=self.dropout,
            ).to(self.device)
        elif self.model_type == "nhits":
            return NHITSModel(
                self.seq_length, self.pred_len,
                n_features=1,
                n_stacks=self.n_stacks,
                n_blocks=self.n_blocks,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _train_epoch(self, model, optimizer, loader) -> float:
        model.train()
        total_loss = 0.0
        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = mape_loss(model(x), y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _run_inference(
        self, model, loader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model on loader. Returns (pred, true, indices)."""
        pred, true, _, indices = collect_predictions(model, loader, self.device, use_log_return=self.use_log_return)
        return pred, true, indices

    def _collect_train_errors(self, model, loader) -> np.ndarray:
        """Compute residuals on training data after final training."""
        _, true, errors, _ = collect_predictions(model, loader, self.device, use_log_return=self.use_log_return)
        return errors

    @torch.no_grad()
    def _collect_decomp_components(self, model, loader) -> Optional[dict]:
        """Collect decomposition component forecasts when supported by model."""
        if not hasattr(model, "get_component_forecasts") or not hasattr(model, "decomposition"):
            return None

        model.eval()
        trend_all, seasonal_all, combined_all = [], [], []
        trend_branch_all, seasonal_branch_all = [], []
        raw_trend_all, raw_seasonal_all, raw_residual_all = [], [], []
        target_all = []
        first_batch = True
        for x, _, _ in loader:
            raw_trend, raw_seasonal, raw_residual = model.decomposition(x.to(self.device))
            raw_trend_all.append(raw_trend.detach().cpu().numpy())
            raw_seasonal_all.append(raw_seasonal.detach().cpu().numpy())
            raw_residual_all.append(raw_residual.detach().cpu().numpy())
            target_all.append(x[:, :, -1].detach().cpu().numpy())

            components = model.get_component_forecasts(x.to(self.device))
            trend_all.append(components["trend"].detach().cpu().numpy())
            seasonal_all.append(components["seasonal"].detach().cpu().numpy())
            combined_all.append(components["combined"].detach().cpu().numpy())

            # Keep explicit branch outputs for one-by-one inspection.
            trend_branch_all.append(components.get("trend", torch.empty(0)).detach().cpu().numpy())
            seasonal_branch_all.append(components.get("seasonal", torch.empty(0)).detach().cpu().numpy())

            # One-time debug prints to inspect RevIN stats and component forecasts
            if first_batch:
                try:
                    logging.debug("\n[DEBG] Decomposition debug — first batch:")
                    # Trend branch RevIN
                    try:
                        tb_rev = model.trend_branch.rev_norm
                        logging.debug("[DEBG] Trend RevIN mean: %s", None if tb_rev.mean is None else tb_rev.mean.detach().cpu().numpy().shape)
                        logging.debug("[DEBG] Trend RevIN std: %s", None if tb_rev.std is None else tb_rev.std.detach().cpu().numpy().shape)
                        if getattr(tb_rev, 'gamma', None) is not None:
                            logging.debug("[DEBG] Trend RevIN gamma: %s", tb_rev.gamma.detach().cpu().numpy().shape)
                            logging.debug("[DEBG] Trend RevIN beta: %s", tb_rev.beta.detach().cpu().numpy().shape)
                    except Exception as e:
                        logging.debug("[DEBG] Trend RevIN inspect failed: %s", e)

                    # Seasonality branch RevIN (if available)
                    try:
                        sb_model = getattr(model.seasonality_branch, 'model', None)
                        if sb_model is not None and hasattr(sb_model, 'rev_norm'):
                            sb_rev = sb_model.rev_norm
                            logging.debug("[DEBG] Seasonal RevIN mean: %s", None if sb_rev.mean is None else sb_rev.mean.detach().cpu().numpy().shape)
                            logging.debug("[DEBG] Seasonal RevIN std: %s", None if sb_rev.std is None else sb_rev.std.detach().cpu().numpy().shape)
                            if getattr(sb_rev, 'gamma', None) is not None:
                                logging.debug("[DEBG] Seasonal RevIN gamma: %s", sb_rev.gamma.detach().cpu().numpy().shape)
                                logging.debug("[DEBG] Seasonal RevIN beta: %s", sb_rev.beta.detach().cpu().numpy().shape)
                        else:
                            logging.debug("[DEBG] Seasonal branch has no RevIN or model attribute")
                    except Exception as e:
                        logging.debug("[DEBG] Seasonal RevIN inspect failed: %s", e)

                    # Component forecast summaries
                    try:
                        t = components['trend']
                        s = components['seasonal']
                        c = components['combined']
                        logging.debug(f"[DEBG] trend forecast mean/std: {t.mean().item():.4f} / {t.std().item():.4f}")
                        logging.debug(f"[DEBG] seasonal forecast mean/std: {s.mean().item():.4f} / {s.std().item():.4f}")
                        logging.debug(f"[DEBG] combined forecast mean/std: {c.mean().item():.4f} / {c.std().item():.4f}")
                    except Exception as e:
                        logging.debug("[DEBG] Component forecast inspect failed: %s", e)
                finally:
                    first_batch = False

        if not trend_all:
            return None

        return {
            "raw_trend": np.concatenate(raw_trend_all).flatten(),
            "raw_seasonal": np.concatenate(raw_seasonal_all).flatten(),
            "raw_residual": np.concatenate(raw_residual_all).flatten(),
            "raw_target": np.concatenate(target_all).flatten(),
            "trend_branch": np.concatenate(trend_branch_all).flatten(),
            "seasonal_branch": np.concatenate(seasonal_branch_all).flatten(),
            "combined": np.concatenate(combined_all).flatten(),
            "aggregation_method": getattr(model.aggregation, "aggregation_method", None),
            "trend_weight": float(model.aggregation.trend_weight.detach().cpu().item()) if hasattr(model.aggregation, "trend_weight") else None,
            "seasonal_weight": float(model.aggregation.seasonal_weight.detach().cpu().item()) if hasattr(model.aggregation, "seasonal_weight") else None,
            # Visualizer expects keys 'trend' and 'seasonal' — provide aliases
            "trend": np.concatenate(trend_branch_all).flatten(),
            "seasonal": np.concatenate(seasonal_branch_all).flatten(),
        }

    def _compute_results(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        train_errors: Optional[np.ndarray] = None,
    ) -> dict:
        metrics = compute_metrics(pred, true)
        tc_min, cs_star, tc_components = sweep_tc(
            pred,
            forecast_errors=train_errors if train_errors is not None else np.zeros_like(pred),
            holding_cost=self.holding_cost,
            lead_time=self.lead_time,
            ordering_cost=self.ordering_cost,
        )
        metrics.update({"TC_min": tc_min, "c_s_star": cs_star, "TC_components": tc_components})
        return metrics

    @abstractmethod
    def _val_metric(self, model, loader) -> float:
        pass


# ── Scenario trainers ─────────────────────────────────────────────────────────

class Scenario1Trainer(BaseTrainer):
    """Validates using MAPE loss."""

    def __init__(
        self,
        seq_length: int = 6,
        ff_dim: Optional[int] = None,
        n_block: Optional[int] = None,
        dropout: Optional[float] = None,
        n_stacks: Optional[int] = None,
        n_layers: Optional[int] = None,
        layer_dim: Optional[int] = None,
        n_blocks: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        pred_len: int = 3,
        batch_size: int = 4,
        lr: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        holding_cost: float = 2.0,
        lead_time: int = 2,
        ordering_cost: float = 50_000,
        seed: int = 42,
        trial: Optional[object] = None,
        visualizer=None,
        model_type: str = "tsmixer",
        # Decomposed model parameters
        use_decomposition: bool = True,
        decomposition_method: str = "stl",
        seasonal_period: int = 4,
        stl_robust: bool = True,
        stl_seasonal: int = 7,
        stl_trend: Optional[int] = None,
        stl_low_pass: Optional[int] = None,
        trend_hidden_dim: Optional[int] = None,
        trend_n_layers: int = 1,
        seasonality_model: str = "tsmixer",
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
        use_log_return: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            seq_length=seq_length,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            patience=patience,
            holding_cost=holding_cost,
            lead_time=lead_time,
            ordering_cost=ordering_cost,
            pred_len=pred_len,
            scenario=1,
            val_metric_type="mape",
            seed=seed,
            trial=trial,
            visualizer=visualizer,
            model_type=model_type,
            ff_dim=ff_dim,
            n_block=n_block,
            dropout=dropout,
            n_stacks=n_stacks,
            n_layers=n_layers,
            layer_dim=layer_dim,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            use_decomposition=use_decomposition,
            decomposition_method=decomposition_method,
            seasonal_period=seasonal_period,
            stl_robust=stl_robust,
            stl_seasonal=stl_seasonal,
            stl_trend=stl_trend,
            stl_low_pass=stl_low_pass,
            trend_hidden_dim=trend_hidden_dim,
            trend_n_layers=trend_n_layers,
            seasonality_model=seasonality_model,
            aggregation_method=aggregation_method,
            learnable_aggregation=learnable_aggregation,
            hierarchical_decomposition=hierarchical_decomposition,
            use_log_return=use_log_return,
            device=device,
        )

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        pred_np, true_np, _, _ = collect_predictions(model, loader, self.device, use_log_return=self.use_log_return)
        if len(pred_np) == 0:
            return float("inf")
        return float(np.mean(np.abs((true_np - pred_np) / (np.abs(true_np) + 1e-8))) * 100)


class Scenario2Trainer(BaseTrainer):
    """Validates using minimum total inventory cost (TC)."""

    def __init__(
        self,
        seq_length: int = 9,
        ff_dim: Optional[int] = None,
        n_block: Optional[int] = None,
        dropout: Optional[float] = None,
        n_stacks: Optional[int] = None,
        n_layers: Optional[int] = None,
        layer_dim: Optional[int] = None,
        n_blocks: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        pred_len: int = 3,
        batch_size: int = 16,
        lr: float = 1e-4,
        epochs: int = 300,
        patience: int = 40,
        holding_cost: float = 2.0,
        lead_time: int = 2,
        ordering_cost: float = 50_000,
        seed: int = 42,
        trial: Optional[object] = None,
        visualizer=None,
        model_type: str = "tsmixer",
        # Decomposed model parameters
        use_decomposition: bool = False,
        decomposition_method: str = "stl",
        seasonal_period: int = 4,
        stl_robust: bool = True,
        stl_seasonal: int = 7,
        stl_trend: Optional[int] = None,
        stl_low_pass: Optional[int] = None,
        trend_hidden_dim: Optional[int] = None,
        trend_n_layers: int = 1,
        seasonality_model: str = "tsmixer",
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
        use_log_return: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            seq_length=seq_length,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            patience=patience,
            holding_cost=holding_cost,
            lead_time=lead_time,
            ordering_cost=ordering_cost,
            pred_len=pred_len,
            scenario=2,
            val_metric_type="tc",
            seed=seed,
            trial=trial,
            visualizer=visualizer,
            model_type=model_type,
            ff_dim=ff_dim,
            n_block=n_block,
            dropout=dropout,
            n_stacks=n_stacks,
            n_layers=n_layers,
            layer_dim=layer_dim,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            use_decomposition=use_decomposition,
            decomposition_method=decomposition_method,
            seasonal_period=seasonal_period,
            stl_robust=stl_robust,
            stl_seasonal=stl_seasonal,
            stl_trend=stl_trend,
            stl_low_pass=stl_low_pass,
            trend_hidden_dim=trend_hidden_dim,
            trend_n_layers=trend_n_layers,
            seasonality_model=seasonality_model,
            aggregation_method=aggregation_method,
            learnable_aggregation=learnable_aggregation,
            hierarchical_decomposition=hierarchical_decomposition,
            use_log_return=use_log_return,
            device=device,
        )

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        pred_np, true_np, errors, _ = collect_predictions(model, loader, self.device, use_log_return=self.use_log_return)
        if len(pred_np) == 0:
            return float("inf")

        tc_min, _, _ = sweep_tc(
            pred_np,
            forecast_errors=errors,
            holding_cost=self.holding_cost,
            lead_time=self.lead_time,
            ordering_cost=self.ordering_cost,
        )
        return tc_min