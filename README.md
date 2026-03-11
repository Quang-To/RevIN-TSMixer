# RevIN-TSMixer

> **Paper:** [RevNI-TSMixer.pdf](RevNI-TSMixer.pdf)

---

## Motivation

Accurate demand forecasting is a critical prerequisite for efficient inventory management, yet it remains challenging in practice due to non-stationary demand patterns and the influence of external macroeconomic drivers. Traditional statistical models struggle to capture complex multivariate dependencies, while many deep learning approaches optimise purely for forecast accuracy — ignoring the downstream inventory costs that ultimately matter to decision-makers.

This work addresses two limitations simultaneously:

1. **Distribution shift.** Monthly demand data exhibits time-varying mean and variance. Without normalisation, models trained on historical distributions generalise poorly to new periods. We apply **Reversible Instance Normalisation (RevIN)**, which removes instance-level statistics before the encoder and re-injects them after the decoder, allowing the model to learn normalised patterns while preserving the original scale for output.

2. **Accuracy ≠ cost optimality.** A model with lower MAPE does not necessarily yield lower inventory total cost. We embed an **EOQ inventory model with safety stock** (Scenario 2) directly into the validation objective, so hyperparameter search explicitly minimises total cost TC — comprising ordering, holding, and shortage costs — rather than a surrogate forecast error.

The backbone architecture is **TSMixer**, a lightweight MLP-Mixer that alternates *time mixing* (capturing temporal patterns across the lookback window) and *feature mixing* (capturing cross-feature dependencies across macroeconomic indicators) without any attention mechanism, making it efficient on small-to-medium tabular time series.

The model is trained and evaluated on a monthly macroeconomic demand dataset for the Vietnamese market, using features such as imports, industrial production index, disbursed FDI, competitor quantity, and promotion spend to forecast product demand up to 3 months ahead.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<Quang-To>/RevNI-TSMixer.git
cd RevNI-TSMixer

# 2. Install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### Train

Open `main.py` and adjust the config block at the top (scenario, hyperparameters, cost parameters), then:

```bash
python main.py
```

### Hyperparameter search (Optuna)

```bash
python -m src.optimizer.optuna
```

Top-3 checkpoints and metric reports are saved to `checkpoints_optuna/`.

### Exploratory Data Analysis

```bash
jupyter notebook src/data/EDA.ipynb
```

---

## Best Results (Scenario 2)

| MAE | RMSE | MAPE | TC\_min | *c*\_s\* |
|-----|------|------|---------|---------|
| 29 652 | 37 136 | 8.54 % | 268 496 | 1.56 |

`seq_length=9`, `n_block=2`, `dropout=0.5`, `batch_size=4`, `ff_dim=64`, `lr=1e-4`

---

## Requirements

Python ≥ 3.9, PyTorch ≥ 2.0 — see [requirements.txt](requirements.txt) for the full list.