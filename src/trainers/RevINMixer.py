import time
import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path

from src.utils.seed import set_seed
from src.utils.visualization import TrainingVisualizer

from src.models.ForecastModel.ForecastModel import ForecastModel
from src.data.dataset import TimeSeriesData
from src.models.InventoryModel.InventoryModel import InventoryModel

# ── Loss ──────────────────────────────────────────────────────────────────────
def mape_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((true - pred) / (torch.abs(true) + 1e-8))) * 100


# ── Shared utils ──────────────────────────────────────────────────────────────
def get_loaders(seq_length, batch_size, pred_len):
    train = TimeSeriesData(seq_length, batch_size, pred_len, "train").get_loader()
    val   = TimeSeriesData(seq_length, batch_size, pred_len, "val").get_loader()
    test  = TimeSeriesData(seq_length, batch_size, pred_len, "test").get_loader()
    return train, val, test

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape)

def sweep_tc(pred: np.ndarray, holding_cost=2, lead_time=2,
             ordering_cost=50_000, n_steps=1000) -> tuple:
    """Sweep c_s ∈ (0, 10] → return (TC_min, c_s*)."""
    pred = np.clip(pred, 1.0, None)
    best_tc, best_cs = float("inf"), None
    for cs in np.linspace(0.01, 10.0, n_steps):
        tc = InventoryModel(cs, holding_cost, lead_time, ordering_cost).total_cost(pred)
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
    return np.concatenate(preds).flatten(), np.concatenate(trues).flatten()


# ── Base Trainer ──────────────────────────────────────────────────────────────

class BaseTrainer:
    def __init__(self, seq_length, ff_dim, dropout, pred_len,
                 n_block, batch_size, lr, epochs, patience,
                 holding_cost, lead_time, ordering_cost, scenario: int = 1):
        self.seq_length    = seq_length
        self.ff_dim        = ff_dim
        self.dropout       = dropout
        self.pred_len      = pred_len
        self.n_block       = n_block
        self.batch_size    = batch_size
        self.lr            = lr
        self.epochs        = epochs
        self.patience      = patience
        self.holding_cost  = holding_cost
        self.lead_time     = lead_time
        self.ordering_cost = ordering_cost
        self.scenario      = scenario
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer    = TrainingVisualizer(save_dir="results")

    def _build_model(self):
        return ForecastModel(self.seq_length, self.ff_dim, self.dropout,
                             self.pred_len, self.n_block).to(self.device)

    def _loss(self, model, x, y) -> torch.Tensor:
        """Training loss — always MAPE for both scenarios."""
        return mape_loss(model(x), y)

    def _val_metric(self, model, loader) -> float:
        """Validation metric — overridden per scenario."""
        raise NotImplementedError

    def _print_results(self, pred, true):
        m = compute_metrics(pred, true)
        tc_min, cs_star = sweep_tc(pred, self.holding_cost,
                                   self.lead_time, self.ordering_cost)
        print(f"\n── Results ──────────────────────────────────────")
        print(f"  MAE   : {m['MAE']:>15.4f}")
        print(f"  MSE   : {m['MSE']:>15.4f}")
        print(f"  RMSE  : {m['RMSE']:>15.4f}")
        print(f"  MAPE  : {m['MAPE']:>14.4f} %")
        print(f"  TC_min: {tc_min:>15.2f}")
        print(f"  c_s*  : {cs_star:>15.4f}")
        return {**m, "TC_min": tc_min, "c_s_star": cs_star}

    def train(self) -> tuple:
        set_seed()
        train_loader, val_loader, test_loader = get_loaders(
            self.seq_length, self.batch_size, self.pred_len
        )
        model     = self._build_model()
        optimizer = Adam(model.parameters(), lr=self.lr)

        best_val, best_state, no_improve, best_epoch = float("inf"), None, 0, 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # train
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self._loss(model, x, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader) if len(train_loader) > 0 else 1

            # validate
            val_metric = self._val_metric(model, val_loader)
            
            # Log metrics for visualization
            self.visualizer.log_epoch(epoch, train_loss, val_metric)

            print(f"Epoch {epoch:>4} | train_mape={train_loss:>8.4f}% | "
                  f"val={val_metric:>10.4f} | {time.time()-t0:.1f}s")

            if val_metric < best_val:
                best_val   = val_metric
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\nEarly stop at epoch {epoch}.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        pred, true = collect_predictions(model, test_loader, self.device)
        metrics = self._print_results(pred, true)
        
        # Generate visualizations
        print("\n" + "="*50)
        print(f"  Generating visualizations for Scenario {self.scenario}...")
        print("="*50)
        self.visualizer.plot_training_history(scenario=self.scenario)
        self.visualizer.plot_predictions_vs_actual(pred, true, scenario=self.scenario)
        self.visualizer.plot_test_metrics(metrics, scenario=self.scenario)
        self.visualizer.plot_comparison_with_baseline(pred, true, scenario=self.scenario)
        self.visualizer.plot_metrics_summary(metrics, epoch=best_epoch, scenario=self.scenario)
        print(f"✓ All plots saved to: results/")
        print("="*50)
        
        return best_val, best_state, metrics

class Scenario1Trainer(BaseTrainer):
    """
    Train: MAPE loss
    Val  : MAPE  ← Optuna/Grid Search minimises this
    """

    def __init__(self, seq_length=6, ff_dim=128, dropout=0.1, pred_len=3,
                 n_block=1, batch_size=4, lr=1e-4, epochs=100, patience=10,
                 holding_cost=2, lead_time=2, ordering_cost=50_000):
        super().__init__(seq_length, ff_dim, dropout, pred_len, n_block,
                         batch_size, lr, epochs, patience,
                         holding_cost, lead_time, ordering_cost, scenario=1)

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        return sum(mape_loss(model(x.to(self.device)), y.to(self.device)).item()
                   for x, y in loader) / len(loader)


class Scenario2Trainer(BaseTrainer):
    """
    Train: MAPE loss  (same as Scenario 1)
    Val  : TC_min  ← Optuna/Grid Search minimises this
    """

    def __init__(self, seq_length=9, ff_dim=128, dropout=0.1, pred_len=3,
                 n_block=2, batch_size=2, lr=1e-4, epochs=1000, patience=100,
                 holding_cost=2, lead_time=2, ordering_cost=50_000):
        super().__init__(seq_length, ff_dim, dropout, pred_len, n_block,
                         batch_size, lr, epochs, patience,
                         holding_cost, lead_time, ordering_cost, scenario=2)

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        all_preds = []
        for x, _ in loader:
            all_preds.append(model(x.to(self.device)).cpu().numpy())
        pred_np = np.clip(np.concatenate(all_preds).flatten(), 1.0, None)
        tc_min, _ = sweep_tc(pred_np, self.holding_cost,
                             self.lead_time, self.ordering_cost)
        return tc_min