# RevIN-TSMixer

> **Decision-aware probabilistic demand forecasting under distribution shift**

RevIN-TSMixer is a research engineering project for forecasting monthly demand and
turning predictive uncertainty into inventory decisions. It combines reversible
instance normalization, lightweight temporal/feature mixing, quantile forecasting,
and an EOQ-inspired inventory cost objective.

The project is intentionally evaluated as a decision system, not only as a
forecasting model:

```text
historical demand + macro features
              │
              ▼
      temporal data pipeline
              │
              ▼
   TSMixer / N-BEATS / N-HiTS
              │
              ▼
     quantile demand forecast
              │
              ▼
       inventory policy + cost
```

## Why this project

Point forecast accuracy is not the final business objective. Under changing
distributions, a model must also produce calibrated uncertainty and support a
decision that balances holding, ordering, and shortage costs.

The repository therefore contains:

- **Forecasting models:** TSMixer, N-BEATS, and N-HiTS.
- **Uncertainty estimation:** multi-quantile pinball loss and calibration metrics.
- **Decision layer:** EOQ-style inventory cost and service-level evaluation.
- **Stress testing:** six calibrated data-generating scenarios.
- **Statistical evaluation:** independent seeds, Oracle-DGP comparisons, confidence
  intervals, paired Wilcoxon tests, and win rates.

## Research questions

1. Does reversible normalization improve robustness to level and distribution shifts?
2. Does optimizing for inventory cost change model selection compared with MAE/RMSE?
3. Which architectures remain reliable under heteroscedastic, heavy-tailed, and
   structural-break scenarios?
4. Can lagged exogenous variables improve probabilistic forecasts without leaking
   future information?

## Dataset and evaluation protocol

The included dataset contains 120 monthly observations with demand and
macroeconomic/exogenous features. The default chronological split is:

| Split | Range | Size |
|---|---:|---:|
| Train | `[0, 84)` | 84 |
| Validation | `[84, 96)` | 12 |
| Test | `[96, 120)` | 24 |

No random shuffling is used across temporal splits. HPO is performed on validation
data; the test window is reserved for final evaluation.

## Reproducible quick start

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall -q src main.py config.py run_simulation_benchmark.py
```

Run the default training/evaluation pipeline:

```powershell
python main.py
```

Generate the calibrated scenarios:

```powershell
python -m src.data.simulate --no_plot
```

After installing the project in editable mode, the same workflows are available
as named commands:

```powershell
python -m pip install -e .
revin-train --dry-run
revin-simulate --no_plot
```

Run the full multi-seed benchmark (compute-intensive):

```powershell
python run_simulation_benchmark.py `
  --models tsmixer,nbeats,nhits `
  --scenarios s1,s2,s3,s4,s5,s6 `
  --n_trials 50 `
  --n_eval_seeds 30
```

Evaluate simulation-selected hyperparameters on the real dataset:

```powershell
python evaluate_standard_dataset.py --models all --seeds 0,1,2
```

## Results

The checked-in reports are generated artifacts, not marketing claims:

- [Simulation benchmark report](results/simulation_benchmark_report.md)
- [Statistical benchmark report](results/statistical_benchmark_paper_report.md)
- [Scenario specifications](results/scenario_specifications.json)

The current benchmark is deliberately honest: N-BEATS and N-HiTS outperform
TSMixer on several simulated scenarios. This makes the repository useful for
studying robustness and decision quality rather than presenting a selectively
chosen winner.

## Repository map

```text
.
├── data/raw_data/              # small source and generated CSV datasets
├── src/data/                   # loading, windows, simulation, walk-forward splits
│   ├── calibration.py         # Fourier/AR calibration and regime-break statistics
│   ├── scenarios.py           # calibrated scenario generation and specifications
│   ├── oracle.py              # oracle forecasts and innovation quantiles
│   ├── validation.py          # Monte Carlo validation and AR baselines
│   └── simulate.py            # backward-compatible simulator CLI facade
├── src/models/                 # TSMixer, N-BEATS, N-HiTS, RevIN, inventory model
├── src/trainers/               # training and early-stopping loops
├── src/hpo/                    # Optuna objectives and search configuration
├── src/utils/                  # metrics, reporting, visualization, reproducibility
├── tests/                      # fast regression tests
├── main.py                     # default train/evaluate entry point
└── run_simulation_benchmark.py # multi-seed HPO and statistical benchmark
```

## Reproducibility notes

- Seeds are set for Python, NumPy, and PyTorch in `src/utils/seed.py`.
- Benchmark tuning seeds and evaluation seeds are kept disjoint.
- All simulation scenarios use explicit parameters and can save Oracle forecasts.
- Results should be interpreted together with the scenario definitions and
  limitations of the 120-month dataset.

## Limitations and next steps

- The real dataset is small, so conclusions should not be generalized without
  external datasets.
- The simulator is calibrated to the included series and cannot replace
  out-of-domain validation.
- The next engineering priorities are stronger public-data baselines, ablations
  for RevIN and cost-aware training, and CPU/edge inference profiling.

## License

No license has been selected yet. Until one is added, please treat the repository
as source-available rather than assuming permission to redistribute or reuse it.
