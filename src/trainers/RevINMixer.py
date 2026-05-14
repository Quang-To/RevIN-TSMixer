import numpy as np
import torch
from torch.optim import Adam
from typing import Optional
from abc import ABC, abstractmethod
import optuna as _optuna

from src.utils.seed import set_seed
from src.utils.metrics import mape_loss, compute_metrics, sweep_tc, collect_predictions
from src.models.ForecastModel.ForecastModel import ForecastModel
from src.models.NBEATSModel.NBEATSModel import NBEATSModel
from src.models.NHITSModel.NHITSModel import NHITSModel
from src.models.Decomposition.DecomposedForecastModel import DecomposedForecastModel, HierarchicalDecomposedModel
from src.data.dataset import TimeSeriesData
from src.data.walk_forward import WalkForwardSplitter

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
        decomposition_method: str = "ma",
        seasonal_period: int = 4,
        trend_hidden_dim: Optional[int] = None,
        trend_n_layers: int = 1,
        seasonality_model: str = "tsmixer",
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.trend_hidden_dim = trend_hidden_dim or 32
        self.trend_n_layers = trend_n_layers
        self.seasonality_model = seasonality_model
        self.aggregation_method = aggregation_method
        self.learnable_aggregation = learnable_aggregation
        self.hierarchical_decomposition = hierarchical_decomposition


    OPTUNA_EPOCHS   = 300
    OPTUNA_PATIENCE = 50

    def crossval_loss_for_optuna(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        min_epochs: int = 20,
    ) -> tuple[float, dict]:
        """
        Run walk-forward cross-validation for Optuna hyperparameter search.
        Each fold is capped at OPTUNA_EPOCHS (300) with early-stop patience
        OPTUNA_PATIENCE (50), regardless of self.epochs / self.patience.

        Returns:
            mean_loss: Mean validation loss across folds.
            metrics:   Dict with fold-level statistics.
        """
        walk_params = {**walk_params, "forecast_horizon": DEFAULT_FORECAST_HORIZON}
        bsz = batch_size or self.batch_size
        splitter = WalkForwardSplitter(**walk_params)

        fold_losses = []

        orig_epochs, orig_patience = self.epochs, self.patience
        self.epochs   = self.OPTUNA_EPOCHS
        self.patience = self.OPTUNA_PATIENCE

        try:
            for fold_idx, split in enumerate(splitter.get_splits()):
                train_loader = self._make_loader(0, split["train_end"], bsz)
                val_loader   = self._make_loader(split["train_end"], split["val_end"], bsz)
                set_seed(self.seed)
                model     = self._build_model()
                optimizer = Adam(model.parameters(), lr=self.lr)

                best_val, _ = self._train_one_fold(
                    model, optimizer, train_loader, val_loader,
                    fold_idx=fold_idx,
                    min_epochs=min_epochs,
                    verbose=verbose,
                )

                fold_losses.append(best_val)

                if verbose:
                    print(f"    [Fold {fold_idx}] best_val={best_val:.4f}")
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
        }
        return mean_loss, metrics

    def train_and_test_with_best_hparams(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        n_epochs: int = 3000,
        patience: int = 100,
    ) -> dict:
        """
        Final training run after Optuna selects best hyperparameters.
        Trains on ALL data before the test window (train + val combined),
        using early stopping monitored on train loss (no separate val set).
        Evaluates once on the held-out test window.
        """
        bsz              = batch_size or self.batch_size
        forecast_horizon = walk_params.get("forecast_horizon", DEFAULT_FORECAST_HORIZON)
        splitter         = WalkForwardSplitter(**walk_params)
        final            = splitter.get_final_test()

        train_loader = self._make_loader(0, final["train_end"], bsz, forecast_horizon)
        test_loader  = self._make_loader(final["test_start"], final["test_end"], bsz, forecast_horizon)

        if verbose:
            print(f"  [Train] [0:{final['train_end']}] — max {n_epochs} epochs, early stop patience={patience}")

        set_seed(self.seed)
        model     = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        best_loss       = float("inf")
        best_epoch_idx  = 1
        no_improve      = 0
        best_state_dict = None

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)

            if train_loss < best_loss:
                best_loss       = train_loss
                best_epoch_idx  = epoch
                no_improve      = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if self.visualizer is not None:
                self.visualizer.log_epoch(epoch, train_loss, None)

            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == n_epochs):
                print(f"  [Train] Epoch {epoch:>4}/{n_epochs} | train={train_loss:.4f} | best={best_loss:.4f}")

            if no_improve >= patience:
                if verbose:
                    print(f"  [Train] Early stop at epoch {epoch}, best_epoch={best_epoch_idx}")
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        train_errors                       = self._collect_train_errors(model, train_loader)
        test_pred, test_true, test_indices = self._run_inference(model, test_loader)
        test_metrics                       = self._compute_results(test_pred, test_true, train_errors)
        decomp_components                  = self._collect_decomp_components(model, test_loader) if self.use_decomposition else None

        if verbose:
            print(f"\n[Result] Epochs={best_epoch_idx} | TC_min={test_metrics['TC_min']:.4f} | MAPE={test_metrics['MAPE']:.4f}%")

        return {
            "metrics":      test_metrics,
            "test_pred":    test_pred,
            "test_true":    test_true,
            "test_indices": test_indices,
            "best_epoch":   best_epoch_idx,
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
                print(f"\n[Fold {fold}] train_end={split['train_end']} | val_end={split['val_end']}")

            train_loader = self._make_loader(0, split["train_end"], bsz)
            val_loader   = self._make_loader(split["train_end"], split["val_end"], bsz)

            set_seed(self.seed)
            model     = self._build_model()
            optimizer = Adam(model.parameters(), lr=self.lr)

            best_val, best_epoch = self._train_one_fold(
                model, optimizer, train_loader, val_loader,
                fold_idx=fold,
                verbose=(verbose >= 2),
            )

            val_folds.append({"fold": fold, "best_val": best_val, "best_epoch": best_epoch})
            best_global_val = min(best_global_val, best_val)

            if verbose >= 1:
                print(f"  → Best val: {best_val:.4f} (epoch {best_epoch})")

        avg_best_epoch = int(round(np.median([f["best_epoch"] for f in val_folds]))) if val_folds else self.epochs
        if verbose >= 1 and val_folds:
            mean_val = np.mean([f["best_val"] for f in val_folds])
            print(f"\n[CV Summary] mean val={mean_val:.4f} | avg_best_epoch={avg_best_epoch}")

        # ── Phase 2: retrain on all pre-test data ─────────────────────────────
        final        = splitter.get_final_test()
        train_loader = self._make_loader(0, final["train_end"], bsz)
        test_loader  = self._make_loader(final["test_start"], final["test_end"], bsz)

        if verbose >= 1:
            print(f"\n[Phase2] Retrain [0:{final['train_end']}] — {avg_best_epoch} epochs")

        set_seed(self.seed)
        model     = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        for epoch in range(1, avg_best_epoch + 1):
            self._train_epoch(model, optimizer, train_loader)

        train_errors                       = self._collect_train_errors(model, train_loader)
        test_pred, test_true, test_indices = self._run_inference(model, test_loader)
        test_metrics                       = self._compute_results(test_pred, test_true, train_errors)

        if verbose >= 1:
            print(f"\n[Test] TC_min={test_metrics['TC_min']:.4f} | MAPE={test_metrics['MAPE']:.4f}%")

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

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_metric = self._val_metric(model, val_loader)

            if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == self.epochs):
                print(f"    [Fold {fold_idx}] Epoch {epoch:>4}/{self.epochs} | train={train_loss:.4f} | val={val_metric:.4f}")

            if val_metric < best_val:
                best_val, best_epoch, no_improve = val_metric, epoch, 0
            else:
                no_improve += 1

            if epoch >= min_epochs and val_metric > best_val * DIVERGE_THRESHOLD:
                if verbose:
                    print(f"    [Fold {fold_idx}] Diverging → prune")
                raise _optuna.exceptions.TrialPruned()

            if no_improve >= self.patience:
                if verbose:
                    print(f"    [Fold {fold_idx}] Early stop epoch {epoch}, best_epoch={best_epoch}")
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
        data = TimeSeriesData(
            seq_length=self.seq_length,
            batch_size=batch_size or self.batch_size,
            pred_length=self.pred_len,
            split_start=split_start,
            split_end=split_end,
            forecast_horizon=forecast_horizon,
        )
        return data.get_loader()

    def _build_model(self):
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
                    trend_hidden_dim=self.trend_hidden_dim,
                    trend_n_layers=self.trend_n_layers,
                    seasonality_model=self.seasonality_model,
                    seasonality_hidden_dim=self.hidden_dim,
                    seasonality_n_blocks=self.n_blocks,
                    seasonality_n_stacks=self.n_stacks,
                    seasonality_n_layers=self.n_layers,
                    seasonality_layer_dim=self.layer_dim,
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
                n_features=1,  # Univariate only (NHITS processes last feature)
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
        pred, true, _, indices = collect_predictions(model, loader, self.device)
        return pred, true, indices

    def _collect_train_errors(self, model, loader) -> np.ndarray:
        """Compute residuals on training data after final training."""
        _, true, errors, _ = collect_predictions(model, loader, self.device)
        return errors

    @torch.no_grad()
    def _collect_decomp_components(self, model, loader) -> Optional[dict]:
        """Collect decomposition component forecasts when supported by model."""
        if not hasattr(model, "get_component_forecasts"):
            return None

        model.eval()
        trend_all, seasonal_all, combined_all = [], [], []
        for x, _, _ in loader:
            components = model.get_component_forecasts(x.to(self.device))
            trend_all.append(components["trend"].detach().cpu().numpy())
            seasonal_all.append(components["seasonal"].detach().cpu().numpy())
            combined_all.append(components["combined"].detach().cpu().numpy())

        if not trend_all:
            return None

        return {
            "trend": np.concatenate(trend_all).flatten(),
            "seasonal": np.concatenate(seasonal_all).flatten(),
            "combined": np.concatenate(combined_all).flatten(),
        }

    def _compute_results(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        train_errors: Optional[np.ndarray] = None,
    ) -> dict:
        metrics = compute_metrics(pred, true)
        tc_min, cs_star = sweep_tc(
            pred,
            forecast_errors=train_errors if train_errors is not None else np.zeros_like(pred),
            holding_cost=self.holding_cost,
            lead_time=self.lead_time,
            ordering_cost=self.ordering_cost,
        )
        metrics.update({"TC_min": tc_min, "c_s_star": cs_star})
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
        decomposition_method: str = "ma",
        seasonal_period: int = 4,
        trend_hidden_dim: Optional[int] = None,
        trend_n_layers: int = 1,
        seasonality_model: str = "tsmixer",
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
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
            trend_hidden_dim=trend_hidden_dim,
            trend_n_layers=trend_n_layers,
            seasonality_model=seasonality_model,
            aggregation_method=aggregation_method,
            learnable_aggregation=learnable_aggregation,
            hierarchical_decomposition=hierarchical_decomposition,
        )

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        return sum(
            mape_loss(model(x.to(self.device)), y.to(self.device)).item()
            for x, y, _ in loader
        ) / len(loader)


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
        decomposition_method: str = "ma",
        seasonal_period: int = 4,
        trend_hidden_dim: Optional[int] = None,
        trend_n_layers: int = 1,
        seasonality_model: str = "tsmixer",
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False,
        hierarchical_decomposition: bool = False,
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
            trend_hidden_dim=trend_hidden_dim,
            trend_n_layers=trend_n_layers,
            seasonality_model=seasonality_model,
            aggregation_method=aggregation_method,
            learnable_aggregation=learnable_aggregation,
            hierarchical_decomposition=hierarchical_decomposition,
        )

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        all_preds, all_trues = [], []
        for x, y, _ in loader:
            all_preds.append(model(x.to(self.device)).cpu().numpy())
            all_trues.append(y.numpy())
        if not all_preds:
            return float("inf")

        pred_np = np.clip(np.concatenate(all_preds).flatten(), PRED_CLIP_MIN, None)
        true_np = np.concatenate(all_trues).flatten()
        errors  = true_np - pred_np

        tc_min, _ = sweep_tc(
            pred_np,
            forecast_errors=errors,
            holding_cost=self.holding_cost,
            lead_time=self.lead_time,
            ordering_cost=self.ordering_cost,
        )
        return tc_min