import numpy as np
import torch
from typing import Optional

from src.utils.metrics import collect_predictions, aggregate_overlapping_forecasts
from src.trainers.base_trainer import BaseTrainer


class Scenario1Trainer(BaseTrainer):
    """Validates using MAPE loss."""

    def __init__(self, seq_length: int = 6, pred_len: int = 3, batch_size: int = 4, lr: float = 1e-4,
                 epochs: int = 100,patience: int = 10, holding_cost: float = 4_000.0, shortage_cost: float = 30_000.0, lead_time: int = 2,
                 ordering_cost: float = 50_000, seed: int = 42, trial: Optional[object] = None,
                 visualizer=None,model_type: str = "tsmixer", use_decomposition: bool = True, device: Optional[torch.device] = None,
                 **kwargs):
        super().__init__(seq_length=seq_length, batch_size=batch_size, lr=lr, epochs=epochs, patience=patience,
                        holding_cost=holding_cost, shortage_cost=shortage_cost, lead_time=lead_time, ordering_cost=ordering_cost,
                        pred_len=pred_len, scenario=1, val_metric_type="mape", seed=seed, trial=trial, visualizer=visualizer,
                        model_type=model_type, use_decomposition=use_decomposition, device=device, **kwargs)

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")
        pred_np, true_np, _, indices, q_mat = collect_predictions(model, loader, self.device, use_log_return=self.use_log_return)
        if len(pred_np) == 0:
            return float("inf")
        pred_np, true_np, _, _ = aggregate_overlapping_forecasts(
            pred_np, true_np, q_mat, indices,
            self.pred_len, self.seq_length, loader.dataset.forecast_horizon,
        )
        return float(np.mean(np.abs((true_np - pred_np) / (np.abs(true_np) + 1e-8))) * 100)
