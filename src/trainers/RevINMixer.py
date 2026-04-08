import numpy as np
import torch
import optuna
from torch.optim import Adam
from typing import Optional

from src.utils.seed import set_seed
from src.utils.visualization import TrainingVisualizer
from src.models.ForecastModel.ForecastModel import ForecastModel
from src.data.dataset import TimeSeriesData
from src.models.InventoryModel.InventoryModel import InventoryModel


def mape_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((true - pred) / (torch.abs(true) + 1e-8))) * 100


def get_loaders(seq_length, batch_size, pred_len):
    train = TimeSeriesData(seq_length, batch_size, pred_len, "train").get_loader()
    val = TimeSeriesData(seq_length, batch_size, pred_len, "val").get_loader()
    test = TimeSeriesData(seq_length, batch_size, pred_len, "test").get_loader()
    return train, val, test


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def sweep_tc(pred: np.ndarray,
             forecast_errors: np.ndarray,
             holding_cost: float = 2.0,
             lead_time: int = 2,
             ordering_cost: float = 50_000,
             n_steps: int = 1000) -> tuple:
    pred = np.clip(pred, 1.0, None)
    best_tc, best_cs = float("inf"), None
    for cs in np.linspace(0.01, 10.0, n_steps):
        tc = InventoryModel(cs, holding_cost, lead_time, ordering_cost).total_cost(
            pred, forecast_errors=forecast_errors
        )
        if np.isfinite(tc) and tc < best_tc:
            best_tc, best_cs = tc, cs
    return best_tc, best_cs


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.numpy())
    pred_array = np.concatenate(preds).flatten()
    true_array = np.concatenate(trues).flatten()
    errors = true_array - pred_array
    return pred_array, true_array, errors


class BaseTrainer:

    def __init__(self, seq_length, ff_dim, dropout, pred_len, n_block,
                 batch_size, lr, epochs, patience,
                 holding_cost, lead_time, ordering_cost,
                 scenario: int = 1, generate_plots: bool = False,
                 val_metric_type: str = "mape", seed: int = 42,
                 trial: Optional[object] = None):
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
        self.visualizer = TrainingVisualizer(save_dir="results") if generate_plots else None
        self.test_pred = None
        self.test_true = None
        self.best_epoch_num = 0
        self.train_errors: Optional[np.ndarray] = None

    def _build_model(self):
        return ForecastModel(self.seq_length, self.ff_dim, self.dropout,
                             self.pred_len, self.n_block).to(self.device)

    def _loss(self, model, x, y) -> torch.Tensor:
        return mape_loss(model(x), y)

    def _val_metric(self, model, loader) -> float:
        raise NotImplementedError

    def _compute_results(self, pred, true):
        """Evaluate trên test set, dùng test errors để tính TC."""
        metrics = compute_metrics(pred, true)
        test_errors = true - pred
        tc_min, cs_star = sweep_tc(
            pred,
            forecast_errors=test_errors,
            holding_cost=self.holding_cost,
            lead_time=self.lead_time,
            ordering_cost=self.ordering_cost,
            n_steps=1000,  # fine search cho final evaluation
        )
        metrics.update({"TC_min": tc_min, "c_s_star": cs_star})
        return metrics

    def train(self) -> tuple:
        set_seed(self.seed)
        train_loader, val_loader, test_loader = get_loaders(
            self.seq_length, self.batch_size, self.pred_len
        )

        model = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        _, train_true, train_errors = collect_predictions(model, train_loader, self.device)
        self.train_errors = train_errors

        best_val, best_state, no_improve, best_epoch = float("inf"), None, 0, 0

        for epoch in range(1, self.epochs + 1):
            # ---- Train ----
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self._loss(model, x, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            _, _, self.train_errors = collect_predictions(model, train_loader, self.device)

            # ---- Validate ----
            val_metric = self._val_metric(model, val_loader)

            if self.visualizer:
                self.visualizer.log_epoch(epoch, train_loss, val_metric)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:>4} | train={train_loss:>7.4f} | val={val_metric:>9.2f}")

            # ---- Optuna Pruning Report ----
            if self.trial is not None:
                self.trial.report(val_metric, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

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

        _, _, self.train_errors = collect_predictions(model, train_loader, self.device)

        self.test_pred, self.test_true, _ = collect_predictions(model, test_loader, self.device)
        metrics = self._compute_results(self.test_pred, self.test_true)
        self.best_epoch_num = best_epoch
        return best_val, best_state, metrics

    def generate_visualizations(self, metrics):
        if not self.visualizer or self.test_pred is None:
            return
        print(f"\n[Scenario {self.scenario}] Generating visualizations...")
        self.visualizer.plot_training_history(scenario=self.scenario)
        self.visualizer.plot_predictions_vs_actual(self.test_pred, self.test_true, scenario=self.scenario)
        self.visualizer.plot_test_metrics(metrics, scenario=self.scenario)
        self.visualizer.plot_comparison_with_baseline(self.test_pred, self.test_true, scenario=self.scenario)
        self.visualizer.plot_metrics_summary(metrics, epoch=self.best_epoch_num, scenario=self.scenario)
        print(f"[OK] Plots saved to results/")


class Scenario1Trainer(BaseTrainer):
    def __init__(self, seq_length=6, ff_dim=128, dropout=0.1, pred_len=3,
                 n_block=1, batch_size=4, lr=1e-4, epochs=100, patience=10,
                 holding_cost=2, lead_time=2, ordering_cost=50_000,
                 generate_plots: bool = False, val_metric_type: str = "mape",
                 seed: int = 42, trial: Optional[object] = None):
        super().__init__(seq_length, ff_dim, dropout, pred_len, n_block,
                         batch_size, lr, epochs, patience,
                         holding_cost, lead_time, ordering_cost,
                         scenario=1, generate_plots=generate_plots,
                         val_metric_type=val_metric_type, seed=seed, trial=trial)

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        if self.val_metric_type == "mape":
            loss_sum = 0.0
            for x, y in loader:
                loss_sum += mape_loss(model(x.to(self.device)), y.to(self.device)).item()
            return loss_sum / len(loader)
        return float("inf")


class Scenario2Trainer(BaseTrainer):
    def __init__(self, seq_length=9, ff_dim=128, dropout=0.1, pred_len=3,
                 n_block=2, batch_size=16, lr=1e-4, epochs=300, patience=40,
                 holding_cost=2, lead_time=2, ordering_cost=50_000,
                 generate_plots: bool = False, val_metric_type: str = "tc",
                 seed: int = 42, trial: Optional[object] = None):
        super().__init__(seq_length, ff_dim, dropout, pred_len, n_block,
                         batch_size, lr, epochs, patience,
                         holding_cost, lead_time, ordering_cost,
                         scenario=2, generate_plots=generate_plots,
                         val_metric_type=val_metric_type, seed=seed, trial=trial)

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        all_preds, all_trues = [], []
        for x, y in loader:
            all_preds.append(model(x.to(self.device)).cpu().numpy())
            all_trues.append(y.numpy())
        pred_np = np.clip(np.concatenate(all_preds).flatten(), 1.0, None)
        true_np = np.concatenate(all_trues).flatten()

        if self.val_metric_type == "tc":
            errors = true_np - pred_np
            tc_min, _ = sweep_tc(
                pred_np,
                forecast_errors=errors,
                holding_cost=self.holding_cost,
                lead_time=self.lead_time,
                ordering_cost=self.ordering_cost,
                n_steps=100,  
            )
            return tc_min
        return float("inf")