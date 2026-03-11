# RevIN-TSMixer

> Demand forecasting with **RevIN + TSMixer** optimized for **inventory cost minimization**

**Paper:** [RevNI-TSMixer.pdf](RevNI-TSMixer.pdf)

---

# Overview

Accurate demand forecasting is essential for effective inventory management, yet real-world demand data often exhibits **non-stationary distributions** and is influenced by **macroeconomic factors**. Traditional statistical models struggle with complex multivariate dependencies, while many deep learning approaches optimise only for prediction accuracy without considering the **inventory costs** that ultimately matter for decision-making.

This project proposes a forecasting framework that integrates:

- **Reversible Instance Normalization (RevIN)** to handle distribution shifts in time series data  
- **TSMixer**, a lightweight MLP-based architecture for efficient temporal and feature mixing  
- **EOQ-based inventory optimisation**, where hyperparameter search directly minimises **total inventory cost (TC)**

The model is trained on a **Vietnamese macroeconomic demand dataset** to forecast product demand up to **3 months ahead**, using features such as imports, industrial production index, disbursed FDI, competitor quantity, and promotion spending.

---

# Key Ideas

## Handling Non-Stationary Demand

Monthly demand data often exhibits changing **mean and variance** across time.  
To improve robustness under distribution shift, we apply **Reversible Instance Normalisation (RevIN)**, which removes instance-level statistics before encoding and restores them after prediction.

## Forecasting for Cost-Optimal Decisions

Lower forecast error does **not necessarily lead to lower inventory cost**.  
To address this gap, we integrate an **Economic Order Quantity (EOQ) model with safety stock** into the validation objective. Hyperparameter optimisation therefore directly minimises **total inventory cost (TC)**, including:

- Ordering cost  
- Holding cost  
- Shortage cost

## Efficient Architecture

The forecasting backbone is **TSMixer**, an MLP-Mixer–style architecture that alternates:

- **Time mixing** — capturing temporal patterns across the lookback window  
- **Feature mixing** — capturing dependencies between macroeconomic indicators  

This design avoids attention mechanisms and remains efficient for **tabular multivariate time series**.

---

# Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/<Quang-To>/RevNI-TSMixer.git
cd RevNI-TSMixer
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Adjust the configuration parameters in `main.py` (scenario settings, model hyperparameters, and cost parameters), then run:

```bash
python main.py
```

---

# Hyperparameter Optimisation (Optuna)

To perform automated hyperparameter search:

```bash
python -m src.optimizer.optuna
```

The **Top-3 checkpoints and evaluation results** will be saved in:

```
checkpoints_optuna/
```

---

# Exploratory Data Analysis

Open the EDA notebook:

```bash
jupyter notebook src/data/EDA.ipynb
```

---

# Best Results (Scenario 2)

| Metric | Value |
|------|------|
| MAE | 29,652 |
| RMSE | 37,136 |
| MAPE | 8.54% |
| TC_min | 268,496 |
| c_s* | 1.56 |

Best configuration:

```
seq_length = 9
n_block = 2
dropout = 0.5
batch_size = 4
ff_dim = 64
learning_rate = 1e-4
```

---

# Project Structure

```
RevNI-TSMixer
│
├── src
│   ├── data
│   │   └── EDA.ipynb
│   │
│   ├── model
│   │   ├── revin.py
│   │   └── tsmixer.py
│   │
│   ├── optimizer
│   │   └── optuna.py
│   │
│   └── utils
│
├── checkpoints_optuna
├── main.py
├── requirements.txt
└── README.md
```

---

# Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  

Full dependency list is available in:

```
requirements.txt
```

---