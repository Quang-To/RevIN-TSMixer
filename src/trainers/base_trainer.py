import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional
from abc import ABC, abstractmethod
import optuna as _optuna
import logging

from src.utils.seed import set_seed
from src.utils.metrics import (
    pinball_loss_torch, compute_metrics, quantile_forecast_metrics,
    compute_inventory_metrics, calibration_coverage,
    collect_predictions, DEFAULT_QUANTILE_LEVELS,
)
from src.models.ForecastModel.TSMixer.ForecastModel import ForecastModel
from src.models.ForecastModel.NBEATSModel.NBEATSModel import NBEATSModel
from src.models.ForecastModel.NHITSModel.NHITSModel import NHITSModel
from src.data.dataset import WindowDataset, MultiWindowDataset
from src.data.data_loader import load_series, load_multivariate, split_indices

DEFAULT_FORECAST_HORIZON = 1
GRAD_CLIP_NORM = 1.0


class BaseTrainer(ABC):
    def __init__(self, seq_length: int = 24, batch_size: int = 16, lr: float = 1e-3,
                 epochs: int = 300, patience: int = 30, holding_cost: float = 4_000.0,
                 shortage_cost: float = 30_000.0, lead_time: int = 1, ordering_cost: float = 50_000,
                 pred_len: int = 1, scenario: int = 2, val_metric_type: str = "pinball",
                 seed: int = 42, trial: Optional[object] = None, visualizer=None,
                 model_type: str = "tsmixer",
                 # TSMixer parameters
                 ff_dim: Optional[int] = None,
                 n_block: Optional[int] = None,
                 dropout: Optional[float] = None,
                 # NBEATS parameters
                 n_stacks: Optional[int] = None,
                 n_blocks: Optional[int] = None,
                 n_layers: Optional[int] = None,
                 layer_dim: Optional[int] = None,
                 # NHITS parameters
                 hidden_dim: Optional[int] = None,
                 revin_affine: bool = True,
                 device: Optional[torch.device] = None,
                 **kwargs):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.lead_time = lead_time
        self.ordering_cost = ordering_cost
        self.pred_len = pred_len
        self.scenario = scenario
        self.val_metric_type = val_metric_type
        self.seed = int(seed)
        self.trial = trial
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = visualizer
        self.model_type = model_type.lower()
        self.revin_affine = revin_affine
        self.use_log_return = kwargs.get("use_log_return", False)
        self.data_file = kwargs.get("data_file", None)

        # General hidden & layers
        self.hidden_dim = hidden_dim or ff_dim or layer_dim or 256
        self.ff_dim = self.hidden_dim
        self.layer_dim = self.hidden_dim
        self.n_block = n_block or n_layers or 3
        self.n_layers = self.n_block
        self.dropout = dropout if dropout is not None else 0.1
        self.n_stacks = n_stacks or 2
        self.n_blocks = n_blocks or 2

        self._prep_data()

    def _prep_data(self):
        y, _ = load_series(self.data_file)
        n = len(y)
        train_end, val_end = split_indices(n)
        scale = float(y[:train_end].mean())
        ys = y / scale

        Xmv, _, _ = load_multivariate(self.data_file, k=6)
        col_std = Xmv[:train_end].std(axis=0)
        col_std[col_std == 0] = 1.0
        Xs = Xmv / col_std

        self.data_dict = {
            "ys": ys,
            "scale": scale,
            "Xs": Xs,
            "target_std": float(col_std[0]),
            "n_features": Xmv.shape[1],
            "train_end": train_end,
            "val_end": val_end,
            "n": n,
        }

    def _make_dataset(self, t_range: tuple[int, int]):
        L = self.seq_length
        H = self.pred_len
        if self.model_type in ("tsmixer", "dlinear"):
            return MultiWindowDataset(self.data_dict["Xs"], L, H, t_range), self.data_dict["target_std"]
        return WindowDataset(self.data_dict["ys"], L, H, t_range), self.data_dict["scale"]

    def _make_loader(self, split_start: int, split_end: int, batch_size: Optional[int] = None,
                     shuffle: bool = False, **kwargs):
        ds, _ = self._make_dataset((split_start, split_end))
        bsz = batch_size or self.batch_size
        return DataLoader(ds, batch_size=bsz, shuffle=shuffle)

    def _build_model(self):
        if self.model_type == "tsmixer":
            return ForecastModel(
                seq_length=self.seq_length, ff_dim=self.hidden_dim,
                dropout=self.dropout, pred_len=self.pred_len, n_block=self.n_layers,
                n_features=self.data_dict["n_features"], revin_affine=self.revin_affine,
            ).to(self.device)
        elif self.model_type == "nbeats":
            return NBEATSModel(
                seq_length=self.seq_length, pred_len=self.pred_len,
                n_stacks=self.n_stacks, n_blocks=self.n_blocks,
                n_layers=self.n_layers, layer_dim=self.hidden_dim,
                dropout=self.dropout, revin_affine=self.revin_affine,
            ).to(self.device)
        elif self.model_type == "nhits":
            return NHITSModel(
                seq_length=self.seq_length, pred_len=self.pred_len,
                n_stacks=self.n_stacks, n_blocks=self.n_blocks,
                hidden_dim=self.hidden_dim, n_layers=self.n_layers,
                dropout=self.dropout, revin_affine=self.revin_affine,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _train_epoch(self, model, optimizer, loader) -> float:
        model.train()
        total_loss = 0.0
        q_tensor = torch.tensor(DEFAULT_QUANTILE_LEVELS, dtype=torch.float32, device=self.device)
        for batch in loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            optimizer.zero_grad()
            q_pred = model(x)
            loss = pinball_loss_torch(q_pred, y, q_tensor)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def evaluate_for_optuna(self, split_params: dict, batch_size: Optional[int] = None,
                            verbose: bool = False, min_epochs: int = 1) -> tuple[float, dict]:
        train_end = split_params.get("train_size", self.data_dict["train_end"])
        val_end = train_end + split_params.get("val_size", self.data_dict["val_end"] - self.data_dict["train_end"])

        bsz = batch_size or self.batch_size
        train_loader = self._make_loader(0, train_end, bsz, shuffle=True)
        val_loader = self._make_loader(train_end, val_end, bsz, shuffle=False)

        set_seed(self.seed)
        model = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        best_val, best_epoch, no_improve = float("inf"), 1, 0
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_loss = self._val_metric(model, val_loader)

            if self.trial is not None and isinstance(self.trial, _optuna.trial.Trial):
                self.trial.report(val_loss, step=epoch)
                if self.trial.should_prune():
                    raise _optuna.exceptions.TrialPruned()

            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        metrics = {
            "val_loss": best_val,
            "best_epoch": best_epoch,
            "fold_best_epochs": [best_epoch],
            "cv_median_best_epoch": best_epoch,
        }
        return best_val, metrics

    def train_with_early_stopping_and_test(self, split_params: dict, batch_size: Optional[int] = None,
                                           verbose: bool = True, max_epochs: int = 300, patience: int = 30) -> dict:
        train_end = split_params.get("train_size", 84)
        val_end = train_end + split_params.get("val_size", 12)
        test_end = val_end + split_params.get("test_size", 24)

        bsz = batch_size or self.batch_size
        train_loader = self._make_loader(0, train_end, bsz, shuffle=True)
        val_loader = self._make_loader(train_end, val_end, bsz, shuffle=False)
        test_ds, descale = self._make_dataset((val_end, test_end))
        test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False)

        set_seed(self.seed)
        model = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        best_val, best_epoch, no_improve = float("inf"), 0, 0
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            val_metric = self._val_metric(model, val_loader)

            if self.visualizer is not None:
                self.visualizer.log_epoch(epoch, train_loss, val_metric)

            if val_metric < best_val:
                best_val = val_metric
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch == 1 or epoch % 50 == 0 or no_improve >= patience or epoch == max_epochs):
                logging.info(f"  Epoch {epoch:>4}/{max_epochs} | train={train_loss:.4f} | val={val_metric:.4f} (best_val={best_val:.4f} at epoch {best_epoch})")

            if no_improve >= patience:
                if verbose:
                    logging.info(f"  [Early Stop] Stopped at epoch {epoch}. Best epoch: {best_epoch} with val_loss={best_val:.4f}")
                break

        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

        # Test evaluation
        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                y = batch[1]
                out = model(x).cpu()
                trues.append(y.numpy().reshape(-1))
                preds.append(out.numpy()[:, 0, :] if out.dim() == 3 else out.numpy())

        yt = np.concatenate(trues) * descale
        yq = np.concatenate(preds) * descale
        p50_idx = len(DEFAULT_QUANTILE_LEVELS) // 2
        yp = yq[:, p50_idx]

        metrics = compute_metrics(yp, yt)
        metrics.update(quantile_forecast_metrics(yq, yt))
        metrics.update(calibration_coverage(yq, yt))

        tc, tc_components = compute_inventory_metrics(
            demand_actual=yt,
            quantile_matrix=yq,
            holding_cost=self.holding_cost,
            shortage_cost=self.shortage_cost,
            ordering_cost=self.ordering_cost,
            lead_time=self.lead_time,
        )
        metrics["TC_min"] = tc
        metrics["Total_Cost"] = tc
        metrics["Fill_Rate"] = tc_components.get("fill_rate", np.nan)
        metrics["Cycle_Service_Level"] = tc_components.get("cycle_service_level", np.nan)
        metrics["ECE"] = metrics.get("mean_calibration_error", np.nan)
        metrics["Cov_P5_P95"] = metrics.get("coverage_P95", 0.0) - metrics.get("coverage_P5", 0.0)
        metrics["Cov_P10_P90"] = metrics.get("coverage_P90", 0.0) - metrics.get("coverage_P10", 0.0)
        metrics["Cov_P25_P75"] = metrics.get("coverage_P75", 0.0) - metrics.get("coverage_P25", 0.0)

        if verbose:
            logging.info(f"\n[Result] Best Epoch={best_epoch} | Best Val Loss={best_val:.4f} | Test MAE={metrics['MAE']:.0f} | Test MAPE={metrics['MAPE']:.2f}% | Test Pinball={metrics['PinballLoss']:.0f} | TC={tc:,.0f}")

        return {
            "metrics": metrics,
            "test_pred": yp,
            "test_true": yt,
            "test_indices": list(range(val_end, test_end)),
            "test_quantiles": yq,
            "best_epoch": best_epoch,
            "best_val": best_val,
        }

    def train_and_test_with_best_hparams(self, walk_params: dict, batch_size: Optional[int] = None,
                                         verbose: bool = True, fixed_epochs: Optional[int] = None, **kwargs) -> dict:
        n_epochs = fixed_epochs if fixed_epochs is not None else 100
        val_end = self.data_dict["val_end"]
        test_end = self.data_dict["n"]

        bsz = batch_size or self.batch_size
        train_ds, descale = self._make_dataset((0, val_end))
        train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
        test_ds, _ = self._make_dataset((val_end, test_end))
        test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False)

        set_seed(self.seed)
        model = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(model, optimizer, train_loader)
            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == n_epochs):
                logging.info(f"  [Train] Epoch {epoch:>4}/{n_epochs} | train={train_loss:.4f}")

        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                y = batch[1]
                out = model(x).cpu()
                trues.append(y.numpy().reshape(-1))
                preds.append(out.numpy()[:, 0, :] if out.dim() == 3 else out.numpy())

        yt = np.concatenate(trues) * descale
        yq = np.concatenate(preds) * descale
        p50_idx = len(DEFAULT_QUANTILE_LEVELS) // 2
        yp = yq[:, p50_idx]

        metrics = compute_metrics(yp, yt)
        metrics.update(quantile_forecast_metrics(yq, yt))
        metrics.update(calibration_coverage(yq, yt))
        tc, tc_comp = compute_inventory_metrics(
            demand_actual=yt, quantile_matrix=yq,
            holding_cost=self.holding_cost, shortage_cost=self.shortage_cost,
            ordering_cost=self.ordering_cost, lead_time=self.lead_time,
        )
        metrics["TC_min"] = tc
        return {
            "metrics": metrics,
            "test_pred": yp,
            "test_true": yt,
            "test_indices": list(range(val_end, test_end)),
            "test_quantiles": yq,
            "best_epoch": n_epochs,
        }

    @abstractmethod
    def _val_metric(self, model, loader) -> float:
        pass
