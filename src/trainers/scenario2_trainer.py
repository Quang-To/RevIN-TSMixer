import numpy as np
import torch
from typing import Optional
from src.utils.metrics import pinball_loss_np, DEFAULT_QUANTILE_LEVELS
from src.trainers.base_trainer import BaseTrainer


class Scenario2Trainer(BaseTrainer):
    def __init__(self, seq_length: int = 24, pred_len: int = 1, batch_size: int = 16, lr: float = 1e-4, 
                 epochs: int = 300, patience: int = 30, holding_cost: float = 4_000.0, shortage_cost: float = 30_000.0,  
                 lead_time: int = 1, ordering_cost: float = 50_000, seed: int = 42, trial: Optional[object] = None,
                 visualizer=None, model_type: str = "tsmixer", device: Optional[torch.device] = None, **kwargs):
        super().__init__(seq_length=seq_length, batch_size=batch_size, lr=lr, epochs=epochs,
                         patience=patience, holding_cost=holding_cost, shortage_cost=shortage_cost,
                         lead_time=lead_time, ordering_cost=ordering_cost, pred_len=pred_len,
                         scenario=2, val_metric_type="pinball", seed=seed, trial=trial, visualizer=visualizer,
                         model_type=model_type, device=device, **kwargs)

    @torch.no_grad()
    def _val_metric(self, model, loader) -> float:
        model.eval()
        if len(loader) == 0:
            return float("inf")

        trues, preds = [], []
        for batch in loader:
            x = batch[0].to(self.device)
            y = batch[1]
            out = model(x).cpu()  # (B, H, Q)
            trues.append(y.numpy().reshape(-1))
            preds.append(out.numpy()[:, 0, :] if out.dim() == 3 else out.numpy())

        yt = np.concatenate(trues)
        yp = np.concatenate(preds)
        return pinball_loss_np(yt, yp, DEFAULT_QUANTILE_LEVELS)
