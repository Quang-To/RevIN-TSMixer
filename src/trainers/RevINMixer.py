import numpy as np
import torch
from torch.optim import Adam
from typing import Optional
from abc import ABC, abstractmethod

from src.utils.seed import set_seed
from src.utils.metrics import mape_loss, compute_metrics, sweep_tc, collect_predictions
from src.models.ForecastModel.ForecastModel import ForecastModel
from src.data.dataset import TimeSeriesData
from src.data.walk_forward import WalkForwardSplitter


class BaseTrainer(ABC):
    """
    Base class for all scenario trainers.
    Handles the full walk-forward training loop.
    """

    # ── FIX 1: __init__ được đưa lên đầu class ───────────────────────────────
    def __init__(
        self,
        seq_length: int,
        ff_dim: int,
        dropout: float,
        pred_len: int,
        n_block: int,
        batch_size: int,
        lr: float,
        epochs: int,
        patience: int,
        holding_cost: float,
        lead_time: int,
        ordering_cost: float,
        scenario: int = 1,
        val_metric_type: str = "mape",
        seed: int = 42,
        trial: Optional[object] = None,
        visualizer=None,
    ):
        self.seq_length = seq_length
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pred_len = pred_len
        self.n_block = n_block
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.holding_cost = holding_cost
        self.lead_time = lead_time
        self.ordering_cost = ordering_cost
        self.scenario = scenario
        self.val_metric_type = val_metric_type
        self.seed = seed
        self.trial = trial
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_errors: Optional[np.ndarray] = None
        self.visualizer = visualizer

    # ── Public API ────────────────────────────────────────────────────────────

    def crossval_loss_for_optuna(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        min_epochs: int = 20,
        trend_window: int = 10,
        tail_window: int = 10,
    ) -> float:
        import optuna as _optuna

        # Luôn dùng forecast_horizon=4
        walk_params = dict(walk_params)
        walk_params["forecast_horizon"] = 4
        bsz = batch_size or self.batch_size
        splitter = WalkForwardSplitter(**walk_params)
        fold_losses = []

        for fold_idx, split in enumerate(splitter.get_splits()):
            # Không overlap: val_loader bắt đầu ngay sau train_loader
            train_loader = self._make_loader(0, split["train_end"], bsz, forecast_horizon=4)
            val_loader   = self._make_loader(split["train_end"], split["val_end"], bsz, forecast_horizon=4)
            set_seed(self.seed)
            model     = self._build_model()
            optimizer = Adam(model.parameters(), lr=self.lr)
            best_val, no_improve = float("inf"), 0
            val_history: list[float] = []

            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch(model, optimizer, train_loader)
                val_metric = self._val_metric(model, val_loader)
                val_history.append(val_metric)

                if epoch % 50 == 0 or epoch == 1 or epoch == self.epochs:
                    print(f"    [Fold {fold_idx}] Epoch {epoch:>3} | train={train_loss:.4f} | val={val_metric:.4f}")

                if epoch >= min_epochs:
                    # Prune nếu relative_slope > 0.01
                    if len(val_history) >= trend_window:
                        window = val_history[-trend_window:]
                        slope  = float(np.polyfit(range(trend_window), window, 1)[0])
                        mean_window = np.mean(window)
                        relative_slope = slope / (abs(mean_window) + 1e-8)
                        if relative_slope > 0.01:
                            if verbose:
                                print(
                                    f"    [Fold {fold_idx}] Epoch {epoch:>3} | "
                                    f"Rising trend detected (relative_slope={relative_slope:.4f}) → prune"
                                )
                            raise _optuna.exceptions.TrialPruned()

                    # Early stopping: không overlap, chỉ tính sau warmup
                    if val_metric < best_val:
                        best_val   = val_metric
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            break

                # Chỉ gọi report/should_prune nếu trial là optuna.Trial
                import optuna as _optuna
                if self.trial is not None and isinstance(self.trial, _optuna.trial.Trial):
                    global_step = fold_idx * self.epochs + epoch
                    self.trial.report(val_metric, step=global_step)
                    if self.trial.should_prune():
                        raise _optuna.exceptions.TrialPruned()

            if len(val_history) >= tail_window:
                fold_loss = float(np.mean(val_history[-tail_window:]))
            elif val_history:
                fold_loss = float(np.mean(val_history))
            else:
                fold_loss = float("inf")

            if verbose:
                print(f"    [Fold {fold_idx}] fold_loss (tail mean) = {fold_loss:.4f}")

            fold_losses.append(fold_loss)

        mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        return mean_loss

    def train_and_test_with_best_hparams(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Dùng sau khi chọn best hyperparams: train trên toàn bộ train+val, test trên test window.
        """
        bsz = batch_size or self.batch_size
        splitter = WalkForwardSplitter(**walk_params)
        final = splitter.get_final_test()
        forecast_horizon = walk_params.get("forecast_horizon", 4)
        train_loader = self._make_loader(0, final["train_end"], bsz, forecast_horizon)
        test_loader  = self._make_loader(final["test_start"], final["test_end"], bsz, forecast_horizon)
        val_loader = self._make_loader(
            int(final["train_end"] * 0.8),
            final["train_end"],
            bsz,
            forecast_horizon
        )

        set_seed(self.seed)
        model = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)
        best_val, best_state, no_improve, best_epoch = float("inf"), None, 0, 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_metric = self._val_metric(model, val_loader)  # val_loader thực sự
            if self.visualizer is not None:
                self.visualizer.log_epoch(epoch, train_loss, val_metric)
            if val_metric < best_val:
                best_val = val_metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        _, _, self.train_errors, _ = collect_predictions(model, train_loader, self.device)
        test_pred, test_true, _, test_indices = collect_predictions(model, test_loader, self.device)
        test_metrics = self._compute_results(test_pred, test_true)

        if verbose:
            print(f"[Best hparams] Test TC_min={test_metrics['TC_min']:.4f} | MAPE={test_metrics['MAPE']:.4f}%")

        return {
            "metrics":      test_metrics,
            "test_pred":    test_pred,
            "test_true":    test_true,
            "test_indices": test_indices,
            "best_val":     best_val,
            "best_epoch":   best_epoch,
        }

    def train_walk_forward(
        self,
        walk_params: dict,
        batch_size: Optional[int] = None,
        verbose: int = 1,
    ) -> dict:
        # Luôn dùng forecast_horizon=4
        walk_params = dict(walk_params)
        walk_params["forecast_horizon"] = 4
        bsz = batch_size or self.batch_size
        splitter = WalkForwardSplitter(**walk_params)

        val_folds: list[dict] = []
        best_global_val   = float("inf")
        best_global_state = None

        # Phase 1: Validation folds
        for split in splitter.get_splits():
            if verbose >= 1:
                print(f"\n[Fold {split['fold']}] train_end={split['train_end']} | val_end={split['val_end']}")

            train_loader = self._make_loader(0, split["train_end"], bsz, forecast_horizon=4)
            val_loader   = self._make_loader(split["train_end"], split["val_end"], bsz, forecast_horizon=4)

            set_seed(self.seed)
            model     = self._build_model()
            optimizer = Adam(model.parameters(), lr=self.lr)

            best_val, best_state, no_improve, best_epoch = float("inf"), None, 0, 0

            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch(model, optimizer, train_loader)
                val_metric = self._val_metric(model, val_loader)

                if verbose >= 2 and epoch in (1, self.epochs // 2, self.epochs):
                    print(f"  Epoch {epoch:>3} | train={train_loss:.4f} | val={val_metric:.4f}")

                if val_metric < best_val:
                    best_val   = val_metric
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            _, _, self.train_errors, _ = collect_predictions(model, train_loader, self.device)

            val_folds.append({
                "fold":       split["fold"],
                "best_val":   best_val,
                "best_epoch": best_epoch,
                "metrics":    {"val_metric": best_val},
            })

            if best_val < best_global_val:
                best_global_val   = best_val
                best_global_state = best_state

            if verbose >= 1:
                print(f"  → Best val: {best_val:.4f} (epoch {best_epoch})")

        # CV summary
        if verbose >= 1 and val_folds:
            mean_val = np.mean([f["best_val"] for f in val_folds])
            print(f"\n[CV Summary] mean val = {mean_val:.4f}")

        # Phase 2: Final test (retrain trên toàn bộ train+val)
        final = splitter.get_final_test()
        train_loader = self._make_loader(0, final["train_end"], bsz, forecast_horizon=4)
        test_loader  = self._make_loader(final["test_start"], final["test_end"], bsz, forecast_horizon=4)

        set_seed(self.seed)
        model = self._build_model()
        # Luôn retrain trên toàn bộ train+val, không dùng best_global_state
        optimizer = Adam(model.parameters(), lr=self.lr)
        best_val, best_state, no_improve, best_epoch = float("inf"), None, 0, 0
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_metric = train_loss  # Không có val_loader, chỉ train trên toàn bộ
            if val_metric < best_val:
                best_val = val_metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        _, _, self.train_errors, _ = collect_predictions(model, train_loader, self.device)
        test_pred, test_true, _, test_indices = collect_predictions(model, test_loader, self.device)
        test_metrics = self._compute_results(test_pred, test_true)

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
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_loader(self, split_start: int, split_end: int, batch_size: Optional[int], forecast_horizon: Optional[int] = None):
        # Luôn ưu tiên forecast_horizon truyền vào, mặc định = 4
        fh = forecast_horizon if forecast_horizon is not None else 4
        data = TimeSeriesData(
            seq_length=self.seq_length,
            batch_size=batch_size or self.batch_size,
            pred_length=self.pred_len,
            split_start=split_start,
            split_end=split_end,
            forecast_horizon=fh,
        )
        return data.get_loader()

    def _build_model(self):
        return ForecastModel(
            self.seq_length, self.ff_dim, self.dropout, self.pred_len, self.n_block
        ).to(self.device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _compute_results(self, pred: np.ndarray, true: np.ndarray) -> dict:
        metrics = compute_metrics(pred, true)
        tc_min, cs_star = sweep_tc(
            pred,
            forecast_errors=self.train_errors if self.train_errors is not None else np.zeros_like(pred),
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
        ff_dim: int = 128,
        dropout: float = 0.1,
        pred_len: int = 3,
        n_block: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        holding_cost: float = 2.0,
        lead_time: int = 2,
        ordering_cost: float = 50_000,
        val_metric_type: str = "mape",
        seed: int = 42,
        trial: Optional[object] = None,
        visualizer=None,
    ):
        super().__init__(
            seq_length, ff_dim, dropout, pred_len, n_block,
            batch_size, lr, epochs, patience,
            holding_cost, lead_time, ordering_cost,
            scenario=1, val_metric_type=val_metric_type, seed=seed, trial=trial, visualizer=visualizer,
        )

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        loss_sum = 0.0
        for x, y, _ in loader:
            loss_sum += mape_loss(model(x.to(self.device)), y.to(self.device)).item()
        return loss_sum / len(loader)


class Scenario2Trainer(BaseTrainer):
    """Validates using minimum total inventory cost (TC)."""

    def __init__(
        self,
        seq_length: int = 9,
        ff_dim: int = 128,
        dropout: float = 0.1,
        pred_len: int = 3,
        n_block: int = 2,
        batch_size: int = 16,
        lr: float = 1e-4,
        epochs: int = 300,
        patience: int = 40,
        holding_cost: float = 2.0,
        lead_time: int = 2,
        ordering_cost: float = 50_000,
        val_metric_type: str = "tc",
        seed: int = 42,
        trial: Optional[object] = None,
        visualizer=None,
    ):
        super().__init__(
            seq_length, ff_dim, dropout, pred_len, n_block,
            batch_size, lr, epochs, patience,
            holding_cost, lead_time, ordering_cost,
            scenario=2, val_metric_type=val_metric_type, seed=seed, trial=trial, visualizer=visualizer,
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

        pred_np = np.clip(np.concatenate(all_preds).flatten(), 1.0, None)
        true_np = np.concatenate(all_trues).flatten()
        errors  = true_np - pred_np

        if self.val_metric_type == "tc":
            tc_min, _ = sweep_tc(
                pred_np,
                forecast_errors=errors,
                holding_cost=self.holding_cost,
                lead_time=self.lead_time,
                ordering_cost=self.ordering_cost,
            )
            return tc_min

        raise ValueError(f"Unknown val_metric_type: {self.val_metric_type!r}")