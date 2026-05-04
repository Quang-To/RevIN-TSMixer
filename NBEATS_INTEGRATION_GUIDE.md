# Hướng dẫn: Thêm NBEATS vào RevNI-TSMixer

## Tổng quan các thay đổi

Bạn đã thêm thành công NBEATS như một lựa chọn model song song với TSMixer hiện có. Hệ thống giờ đây hỗ trợ:

- **TSMixer**: Model mixer-based hiện tại
- **NBEATS**: Neural Basis Expansion Analysis (mới thêm)

## Các file đã thay đổi

### 1. **Model Architecture**
- `src/models/NBEATSModel/NBEATSModel.py` (NEW)
  - Implement NBEATS model
  - Sử dụng basis expansion blocks
  - Hỗ trợ stack của blocks cho forecasting

### 2. **Configuration**
- `src/hpo/config.py` (UPDATED)
  - Thêm `SEARCH_SPACE_NBEATS` riêng
  - Thêm `VALID_MODELS` set
  - Giữ `SEARCH_SPACE_TSMIXER` để backward compatibility

### 3. **Trainer Factory**
- `src/hpo/trainer_factory.py` (UPDATED)
  - Thêm `model_type` parameter
  - Support instantiate cả TSMixer và NBEATS trainers
  - Routing parameters dựa trên model type

### 4. **Base Trainer**
- `src/trainers/RevINMixer.py` (UPDATED)
  - `BaseTrainer.__init__`: Thêm `model_type` và tham số NBEATS
  - `_build_model()`: Support cả hai model types
  - `Scenario1Trainer`: Thêm `model_type` parameter
  - `Scenario2Trainer`: Thêm `model_type` parameter

### 5. **Hyperparameter Optimization**
- `src/hpo/objective.py` (UPDATED)
  - `sample_params()`: Chấp nhận `model_type` parameter
  - Dynamic search space selection

- `src/hpo/optuna.py` (UPDATED)
  - `OptunaOptimizer.__init__()`: Thêm `model_type` parameter
  - `_objective()`: Truyền `model_type` cho trainer
  - `_save_top3()`: Lưu model type vào checkpoint
  - Database separation per model type

### 6. **Main Pipeline**
- `main.py` (UPDATED)
  - Thêm `MODEL` configuration variable
  - Model-specific parameter initialization
  - Conditional trainer instantiation

## Cách sử dụng

### Option 1: HPO với TSMixer (default)
```python
from src.hpo.optuna import OptunaOptimizer

optimizer = OptunaOptimizer(
    scenario=2,
    val_metric="tc",
    n_trials=100,
    n_jobs=4,
    model_type="tsmixer"  # <- Explicit
)
optimizer.run()
```

### Option 2: HPO với NBEATS
```python
optimizer = OptunaOptimizer(
    scenario=2,
    val_metric="tc",
    n_trials=100,
    n_jobs=4,
    model_type="nbeats"  # <- Change to NBEATS
)
optimizer.run()
```

### Option 3: Training với best params từ NBEATS
Trong `main.py`, thay đổi:
```python
SCENARIO   = 2
MODEL      = "nbeats"  # <- Change from "tsmixer"
VAL_METRIC = "tc"
SEED       = 42

# NBEATS specific
N_STACKS   = 3
N_LAYERS   = 4
LAYER_DIM  = 128
```

Rồi chạy:
```bash
python main.py
```

## Hyperparameter Search Spaces

### TSMixer Parameters
```python
{
    "seq_length": [1, 2, ..., 12],
    "n_block":    [1, 2, 3],
    "dropout":    [0.1, 0.3, 0.5, 0.7, 0.9],
    "batch_size": [2, 3, 4],
    "ff_dim":     [8, 16, 32, 64, 128],
    "lr":         [1e-4, 1e-5],
}
```

### NBEATS Parameters
```python
{
    "seq_length": [1, 2, ..., 12],
    "n_stacks":   [2, 3, 4],
    "n_layers":   [2, 3, 4],
    "layer_dim":  [64, 128, 256],
    "dropout":    [0.1, 0.3, 0.5],
    "batch_size": [2, 3, 4],
    "lr":         [1e-4, 1e-5],
}
```

## Output Structure

Checkpoints được lưu với naming scheme:
- TSMixer: `s{scenario}_{model_type}_rank{rank}.pt`
- NBEATS: `s{scenario}_{model_type}_rank{rank}.pt`

Ví dụ:
- `s2_tsmixer_rank1.pt` - Best TSMixer trial
- `s2_nbeats_rank1.pt` - Best NBEATS trial

Database Optuna:
- `optuna_s{scenario}_{model_type}.db`

Ví dụ:
- `optuna_s2_tsmixer.db`
- `optuna_s2_nbeats.db`

## Kiến trúc NBEATS được implement

### NBEATSBlock
- Multi-layer fully connected stack
- Basis expansion cho forecast và backcast
- Per-feature processing

### NBEATSModel
- Stacks của NBEATSBlocks
- RevINNorm normalization
- Residual connections

## Comparisons & Tips

| Aspect | TSMixer | NBEATS |
|--------|---------|--------|
| Computation | ~Medium | ~Slower (do basis expansion) |
| Memory | ~Lower | ~Higher |
| Best for | Long sequences | Various horizons |
| Interpretability | Medium | Low |

## Troubleshooting

### Model không load?
```python
# Đảm bảo model_type match giữa checkpoint và code
checkpoint = torch.load("s2_nbeats_rank1.pt")
# Use model_type="nbeats" khi instantiate trainer
```

### HPO chậm?
- Giảm `n_trials`
- Tăng `n_jobs` nếu có multi-GPU
- Sử dụng `patience` nhỏ hơn

### Memory issue?
- Giảm `layer_dim` cho NBEATS
- Giảm `batch_size`
- Sử dụng model_type="tsmixer" (ít memory hơn)

## Next Steps

1. Chạy HPO cho NBEATS: `OptunaOptimizer(..., model_type="nbeats").run()`
2. So sánh kết quả TSMixer vs NBEATS trên cùng dataset
3. Điều chỉnh hyperparameter space nếu cần
4. Thêm model ensemble nếu muốn kết hợp cả hai
