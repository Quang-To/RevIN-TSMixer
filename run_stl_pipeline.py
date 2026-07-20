import os
import sys
import logging
import torch
import numpy as np

# Set up paths to import from the project
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessing import Preprocessing
from src.trainers.RevINMixer import Scenario2Trainer
from src.utils.seed import set_seed
from src.utils.metrics import sweep_tc

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 1. Load data
    data = Preprocessing().preprocess()
    y_full = data["Quantity"].values[:120]  # Take first 120 points
    
    # Define Walk-Forward parameters matching checkpoints
    walk_params = dict(
        seq_length=11,
        pred_length=3,
        forecast_horizon=4,
        train_ratio=0.45,
        val_size=29,
        test_size=29,
        step=3,
    )
    
    trainer = Scenario2Trainer(
        seq_length=11,
        pred_len=3,
        batch_size=3,
        lr=0.0001,
        epochs=3000,
        patience=100,
        holding_cost=2.0,
        lead_time=2,
        ordering_cost=50000.0,
        seed=42,
        model_type="nhits",
        use_decomposition=True,
        decomposition_method="stl",
        seasonal_period=4,
        stl_robust=False,
        stl_seasonal=7,
        stl_trend=41,
        trend_hidden_dim=32,
        trend_n_layers=1,
        seasonality_model="nhits",
        aggregation_method="sum",
        device=device
    )
    
    # Run walk-forward cross validation
    logging.info("Starting Walk-Forward Cross Validation...")
    cv_result = trainer.train_walk_forward(walk_params, verbose=1)
    
    print("\n" + "="*50)
    print(f"{'CROSS-VALIDATION RESULT':^50}")
    print("="*50)
    for fold in cv_result["val_folds"]:
        print(f"Fold {fold['fold']}: Best Val Cost = {fold['best_val']:.2f} (Epoch {fold['best_epoch']})")
    print(f"CV Best Global Val Cost: {cv_result['test']['best_val']:.2f}")
    print(f"CV Average Best Epoch  : {cv_result['test']['best_epoch']}")
    print("="*50 + "\n")
    
    # Train final evaluation (Train on Train+Val, evaluate on Test)
    logging.info("Starting Final Evaluation...")
    final_result = trainer.train_and_test_with_best_hparams(
        walk_params=walk_params,
        batch_size=3,
        verbose=True,
        n_epochs=3000,
        patience=100,
        fold_best_epochs=[fold["best_epoch"] for fold in cv_result["val_folds"]]
    )
    
    print("\n" + "="*50)
    print(f"{'FINAL TEST METRICS':^50}")
    print("="*50)
    print(f"Best Epoch : {final_result['best_epoch']}")
    print(f"MAPE       : {final_result['metrics']['MAPE']:.4f}%")
    print(f"TC_min     : {final_result['metrics']['TC_min']:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
