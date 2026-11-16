import numpy as np
import torch
from typing import Optional
from src.models.InventoryModel.InventoryModel import InventoryModel, DEFAULT_QUANTILE_LEVELS


# ── PyTorch Loss ─────────────────────────────────────────────────────────────

def pinball_loss_torch(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """pred: (B, H, Q) hoặc (B, Q), target: (B, H) hoặc (B,), quantiles: (Q,)."""
    if pred.dim() == 2 and target.dim() == 1:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    target = target.unsqueeze(-1)
    errors = target - pred
    q = quantiles.view(1, 1, -1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    return loss.mean()


def pinball_loss(quantile_preds: torch.Tensor, targets: torch.Tensor, quantile_levels: tuple = DEFAULT_QUANTILE_LEVELS) -> torch.Tensor:
    taus = torch.tensor(quantile_levels, dtype=quantile_preds.dtype, device=quantile_preds.device)
    return pinball_loss_torch(quantile_preds, targets, taus)


def mape_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((true - pred) / (torch.abs(true) + 1e-8))) * 100


# ── NumPy Metrics ────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / (denom + 1e-8)) * 100)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    return all_metrics(y_true, y_pred)


def pinball_loss_np(y_true: np.ndarray, y_pred_q: np.ndarray, quantiles=DEFAULT_QUANTILE_LEVELS) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    q = np.asarray(quantiles, dtype=np.float64).reshape(1, -1)
    errors = y_true - y_pred_q
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    return float(loss.mean())


def quantile_forecast_metrics(
    quantile_matrix: np.ndarray,
    y_true: np.ndarray,
    quantile_levels: tuple = DEFAULT_QUANTILE_LEVELS,
) -> dict:
    q = np.asarray(quantile_matrix, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    n = min(len(y), len(q))
    q, y = q[:n], y[:n]
    losses = {}
    values = []
    for i, tau in enumerate(quantile_levels):
        err = y - q[:, i]
        value = float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))
        losses[f"Pinball_P{int(round(tau * 100))}"] = value
        values.append(value)
    losses["Pinball_Loss"] = float(np.mean(values))
    losses["PinballLoss"] = losses["Pinball_Loss"]
    return losses


def calibration_coverage(
    quantile_matrix: np.ndarray,
    y_true: np.ndarray,
    quantile_levels: tuple = DEFAULT_QUANTILE_LEVELS,
) -> dict:
    levels = np.asarray(quantile_levels)
    qm = np.asarray(quantile_matrix)
    yt = np.asarray(y_true).flatten()
    n = min(len(yt), len(qm))
    qm, yt = qm[:n], yt[:n]

    coverage = {}
    errors = []
    for i, tau in enumerate(levels):
        actual_coverage = float(np.mean(yt <= qm[:, i]))
        lbl = f"P{int(tau*100)}"
        coverage[f"coverage_{lbl}"] = actual_coverage
        coverage[f"expected_{lbl}"] = float(tau)
        errors.append(abs(actual_coverage - tau))

    coverage["mean_calibration_error"] = float(np.mean(errors))
    coverage["ECE"] = coverage["mean_calibration_error"]
    return coverage


def compute_inventory_metrics(
    demand_actual: np.ndarray,
    quantile_matrix: np.ndarray,
    holding_cost: float = 4_000.0,
    shortage_cost: float = 30_000.0,
    ordering_cost: float = 50_000.0,
    lead_time: int = 1,
    quantile_levels: tuple = DEFAULT_QUANTILE_LEVELS,
) -> tuple[float, dict]:
    inv = InventoryModel(
        holding_cost   = holding_cost,
        shortage_cost  = shortage_cost,
        ordering_cost  = ordering_cost,
        lead_time      = lead_time,
        quantile_levels= quantile_levels,
    )
    return inv.total_cost(demand_actual, np.asarray(quantile_matrix, dtype=np.float64))


def aggregate_overlapping_forecasts(pred_p50: np.ndarray, true: np.ndarray, quantile_matrix: np.ndarray, sample_indices: np.ndarray, pred_len: int, seq_length: int, forecast_horizon: int,) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pred_len == 1:
        return pred_p50, true, quantile_matrix, sample_indices
    offset = seq_length + forecast_horizon - 1
    target_indices = np.concatenate([
        np.arange(int(i) + offset, int(i) + offset + pred_len)
        for i in np.asarray(sample_indices).reshape(-1)
    ])
    n = min(len(target_indices), len(true), len(quantile_matrix))
    if n == 0:
        empty = np.array([])
        return empty, empty, np.empty((0, quantile_matrix.shape[-1])), empty
    target_indices = target_indices[:n]
    true = np.asarray(true)[:n]
    quantile_matrix = np.asarray(quantile_matrix)[:n]
    unique = np.unique(target_indices)
    q_agg = np.stack([quantile_matrix[target_indices == i].mean(axis=0) for i in unique])
    true_agg = np.array([true[target_indices == i][0] for i in unique])
    pred_agg = q_agg[:, q_agg.shape[1] // 2]
    return pred_agg, true_agg, q_agg, unique


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device,
    use_log_return: bool = False,
) -> tuple:
    model.eval()
    preds, trues, indices = [], [], []

    for batch in loader:
        if len(batch) == 3:
            x, y, meta = batch
            indices.append(meta[:, 0].numpy())
        else:
            x, y = batch
            indices.append(np.arange(len(y)))

        out = model(x.to(device)).cpu().numpy()
        preds.append(out)
        trues.append(y.numpy())

    if not preds:
        empty = np.array([])
        return empty, empty, empty, empty, empty.reshape(0, len(DEFAULT_QUANTILE_LEVELS))

    q_all = np.concatenate(preds, axis=0)        # (N, H, Q)
    t_all = np.concatenate(trues, axis=0)        # (N, H)
    idx_array = np.concatenate(indices).flatten()

    n_q = q_all.shape[-1]
    p50_idx = n_q // 2

    pred_p50 = q_all[:, :, p50_idx].flatten()
    true_flat = t_all.flatten()
    q_flat = q_all.reshape(-1, n_q)

    target_scale = float(getattr(loader.dataset, "target_scale", 1.0))
    pred_p50 = pred_p50 * target_scale
    true_flat = true_flat * target_scale
    q_flat = q_flat * target_scale

    errors = true_flat - pred_p50
    return pred_p50, true_flat, errors, idx_array, q_flat
