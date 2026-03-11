# RevIN-TSMixer: Inventory-Optimised Demand Forecasting with Reversible Instance Normalisation and TSMixer

## Overview

This repository contains the implementation of **RevIN-TSMixer**, a multivariate time-series demand-forecasting framework that integrates:

* **TSMixer** — an MLP-Mixer architecture that alternates temporal mixing and feature mixing across the input window, capturing both time-step and cross-feature dependencies without attention mechanisms.
* **RevIN** (Reversible Instance Normalisation) — a learnable normalisation layer with affine parameters that reverses distribution shift, applied symmetrically before the encoder and after the decoder head.
* **EOQ-based Inventory Model** — an Economic Order Quantity model with safety stock (using the Normal-loss function) that converts demand forecasts into inventory total cost (TC), enabling direct cost-oriented hyperparameter selection.

The model is evaluated on a macroeconomic demand dataset for the Vietnamese market and is compared under two optimisation scenarios:

| Scenario | Training loss | Validation / HPO objective |
|----------|--------------|----------------------------|
| **S1** | MAPE | MAPE ↓ |
| **S2** | MAPE | TC\_min ↓ (inventory total cost swept over shortage cost *c*\_s) |

---

## Repository Structure

```
RevNI-TSMixer/
├── data/
│   └── raw_data/
│       └── data_TSI_v2.csv          # Monthly macroeconomic + demand data
├── checkpoints_optuna/              # Saved .pt checkpoints and .txt reports
│   ├── s2_rank1.pt / s2_rank1.txt
│   ├── s2_rank2.pt / s2_rank2.txt
│   └── s2_rank3.pt / s2_rank3.txt
├── src/
│   ├── data/
│   │   ├── data_loader.py           # CSV reader
│   │   ├── preprocessing.py         # Feature selection & train/val/test split
│   │   ├── dataset.py               # PyTorch Dataset + DataLoader wrapper
│   │   └── EDA.ipynb                # Exploratory data analysis notebook
│   ├── models/
│   │   ├── ForecastModel/
│   │   │   ├── ForecastModel.py     # RevIN + MixerBlocks + projection head
│   │   │   ├── RevINNorm/
│   │   │   │   └── RevINNorm.py     # Reversible Instance Normalisation
│   │   │   ├── MixingLayer/
│   │   │   │   ├── MixerBlock.py    # Single TimeMixing + FeatureMixing block
│   │   │   │   ├── TimeMixingLayer.py
│   │   │   │   └── FeatureMixingLayer.py
│   │   │   └── TemporalProjectionLayer/
│   │   │       └── TemporalProjectionLayer.py  # Linear seq_len → pred_len
│   │   └── InventoryModel/
│   │       └── InventoryModel.py    # EOQ + safety stock + total cost
│   ├── trainers/
│   │   └── RevINMixer.py            # BaseTrainer, Scenario1Trainer, Scenario2Trainer
│   ├── optimizer/
│   │   └── optuna.py                # Optuna HPO launcher (saves top-3 checkpoints)
│   └── utils/
│       └── seed.py                  # Global reproducibility seed
├── main.py                          # Entry point — edit config block, then run
├── requirements.txt
└── README.md
```

---

## Model Architecture

```
Input  (B, T, F)
   │
   ▼
RevINNorm  ──  norm(x)              # subtract instance mean, divide by std
   │
   ▼
MixerBlock × n_block
   ├─ TimeMixingLayer               # BN → Linear(T→T) per feature + residual
   └─ FeatureMixingLayer            # BN → MLP(F→ff_dim→F) per time-step + residual
   │
   ▼
TemporalProjectionLayer             # Linear(T → pred_len) applied feature-wise
   │
   ▼
RevINNorm  ──  denorm(x)            # restore original scale via stored mean/std
   │
   ▼
Output  (B, pred_len)               # forecast for target feature (Quantity)
```

**Dimensions:** `B` = batch size, `T` = `seq_length` (lookback window), `F` = 6 features.

---

## Dataset

`data_TSI_v2.csv` contains **monthly** records with the following columns:

| Column | Description |
|--------|-------------|
| `Month` | Year-month index (YYYY-MM) |
| `Quantity` | Target — monthly demand (**units**) |
| `CompetitorQuantity` | Competitor sales volume |
| `PromotionAmount` | Marketing / promotion spend (VND) |
| `Construction` | Construction activity index |
| `CPI` | Consumer Price Index (MoM %) |
| `Exports` | Total exports (USD million) |
| `Imports` | Total imports (USD million) |
| `IPI` | Industrial Production Index (MoM %) |
| `RegisteredFDI` | Registered Foreign Direct Investment |
| `DisbursedFDI` | Disbursed FDI |
| `RetailSales` | Retail sales index |

After preprocessing, 6 features are retained: `Imports`, `IPI`, `DisbursedFDI`, `CompetitorQuantity`, `PromotionAmount`, `Quantity`.

**Split:** 80 % train / 10 % validation / 10 % test (chronological, no shuffling of the time axis).

---

## Inventory Model

The `InventoryModel` class implements the **EOQ model with shortage cost and safety stock**:

$$Q^* = \sqrt{\frac{2 K \mu}{h}}$$

$$z^* = \Phi^{-1}\!\left(1 - \frac{h Q^*}{c_s \mu}\right), \quad SS = z^* \sigma \sqrt{L}$$

$$TC = \frac{K\mu}{Q^*} + h\!\left(\frac{Q^*}{2} + SS\right) + \frac{c_s \mu}{Q^*}\,\sigma\sqrt{L}\,\psi(z^*)$$

where $K$ = ordering cost, $h$ = holding cost, $c_s$ = shortage cost, $L$ = lead time, $\mu$ / $\sigma$ = mean / std of the forecast, and $\psi(\cdot)$ is the standard-Normal loss function.

In **Scenario 2**, the shortage cost $c_s$ is swept over $(0, 10]$ to find the $c_s^*$ that minimises $TC$. The resulting $TC_{\min}$ is used as the Optuna validation objective.

---

## Hyperparameter Search Space (Optuna TPE)

| Parameter | Values |
|-----------|--------|
| `seq_length` | 1–9 |
| `n_block` | 1, 2, 3 |
| `dropout` | 0.1, 0.3, 0.5, 0.7, 0.9 |
| `batch_size` | 1, 2, 3, 4 |
| `ff_dim` | 8, 16, 32, 64, 128 |
| `lr` | 1e-4, 1e-5 |

Top-3 trials per scenario are retrained from scratch and saved as `.pt` checkpoints with accompanying `.txt` metric reports in `checkpoints_optuna/`.

---

## Best Results (Scenario 2 — Rank 1)

| Metric | Value |
|--------|-------|
| MAE | 29 652 |
| RMSE | 37 136 |
| MAPE | 8.54 % |
| TC\_min | 268 496 |
| *c*\_s\* | 1.56 |

Hyperparameters: `seq_length=9`, `n_block=2`, `dropout=0.5`, `batch_size=4`, `ff_dim=64`, `lr=1e-4`.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<Quang-To>/RevNI-TSMixer.git
cd RevNI-TSMixer
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional):** If you want CUDA acceleration, install the matching PyTorch wheel from [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) before running the command above.

### 4. Configure and train the model

Open `main.py` and edit the **Configuration** block at the top of the file:

```python
# Select scenario:
#   1 → optimise MAPE
#   2 → optimise inventory total cost (TC_min)
SCENARIO = 2

# Model hyperparameters
SEQ_LENGTH   = 9        # lookback window (number of past months)
N_BLOCK      = 2        # number of MixerBlocks
DROPOUT      = 0.5      # dropout rate
FF_DIM       = 64       # hidden dimension of FeatureMixingLayer
PRED_LEN     = 3        # forecast horizon (months ahead)

# Training hyperparameters
BATCH_SIZE   = 4
LR           = 1e-4     # learning rate
EPOCHS       = 1000
PATIENCE     = 100      # early-stopping patience

# Inventory cost parameters
HOLDING_COST  = 2
LEAD_TIME     = 2
ORDERING_COST = 50_000
```

Then run:

```bash
python main.py
```

The script will train the model and print a results summary:

```
==================================================
  SCENARIO 2 — TRAINING COMPLETE
==================================================
  Best val TC_min      :        262881.6456
--------------------------------------------------
  TEST METRICS
--------------------------------------------------
  MAE                  :         29652.1777
  MSE                  :    1379066752.0000
  RMSE                 :         37135.7891
  MAPE (%)             :             8.5361 %
  TC_min               :         268496.37
  c_s*                 :            1.5600
==================================================
```

### 5. Run Optuna hyperparameter search

```bash
python -m src.optimizer.optuna
```

Checkpoints and metric reports for the top-3 trials of each scenario are saved to `checkpoints_optuna/`.

### 6. Exploratory Data Analysis

```bash
jupyter notebook src/data/EDA.ipynb
```

---

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* See [requirements.txt](requirements.txt) for the full dependency list.

---

## License

This project is released for academic research purposes. Please cite appropriately if you use this code or dataset.