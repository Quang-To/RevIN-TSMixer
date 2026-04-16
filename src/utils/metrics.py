import numpy as np
import torch
from typing import Optional

from src.models.InventoryModel.InventoryModel import InventoryModel


# ── Loss ──────────────────────────────────────────────────────────────────────

def mape_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """MAPE loss for training (returns a scalar tensor)."""
    return torch.mean(torch.abs((true - pred) / (torch.abs(true) + 1e-8))) * 100


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Return MAE, MSE, RMSE, MAPE for a prediction array."""
    mae  = np.mean(np.abs(y_true - y_pred))
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


# ── Inventory cost sweep ──────────────────────────────────────────────────────

def sweep_tc(pred: np.ndarray, forecast_errors: np.ndarray, holding_cost: float = 2.0,
    lead_time: int = 2, ordering_cost: float = 50_000, n_steps: int = 1000,) -> tuple[float, Optional[float]]:
    pred = np.clip(pred, 1.0, None)
    best_tc, best_cs = float("inf"), None
    for cs in np.linspace(0.01, 10.0, n_steps):
        tc = InventoryModel(cs, holding_cost, lead_time, ordering_cost).total_cost(
            pred, forecast_errors=forecast_errors
        )
        if np.isfinite(tc) and tc < best_tc:
            best_tc, best_cs = tc, cs
    return best_tc, best_cs


# ── Prediction collector ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple:
    """
    Run model inference over a DataLoader.
    Returns (pred_array, true_array, errors, idx_array).
    All arrays are 1-D and flattened.
    """
    model.eval()
    preds, trues, indices = [], [], []

    for batch in loader:
        if len(batch) == 3:
            x, y, idx = batch
            indices.append(idx.numpy())
        else:
            x, y = batch
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.numpy())

    if not preds:
        empty = np.array([])
        return empty, empty, empty, empty

    pred_array = np.concatenate(preds).flatten()
    true_array = np.concatenate(trues).flatten()
    idx_array  = np.concatenate(indices).flatten() if indices else np.array([])
    errors     = true_array - pred_array

    return pred_array, true_array, errors, idx_array