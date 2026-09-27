import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    """Visualize training metrics and predictions."""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track metrics during training
        self.train_losses: List[float] = []
        self.val_metrics: List[float] = []
        self.epochs: List[int] = []
    
    def log_epoch(self, epoch: int, train_loss: float, val_metric: float):
        """Log metrics for each epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_metrics.append(val_metric)
    
    def plot_training_history(self, scenario: int = 1):
        """Plot training and validation metrics over epochs."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training Loss
        axes[0].plot(self.epochs, self.train_losses, 'o-', linewidth=2, 
                     markersize=6, color='#2E86AB', label='Train Loss (MAPE)')
        axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('MAPE Loss (%)', fontsize=11, fontweight='bold')
        axes[0].set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Validation Metric
        metric_name = 'TC_min (cost units)' if scenario == 2 else 'MAPE (%)'
        axes[1].plot(self.epochs, self.val_metrics, 'o-', linewidth=2, 
                     markersize=6, color='#A23B72', label='Validation Metric')
        axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1].set_ylabel(metric_name, fontsize=11, fontweight='bold')
        axes[1].set_title('Validation Metric Over Epochs', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        save_path = self.save_dir / f'scenario_{scenario}_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()

    def plot_fan_chart(self, quantiles, true, indices=None, scenario: int = 1):
        """Plot test forecasts with central 50%, 80%, and 90% intervals."""
        q = np.asarray(quantiles, dtype=float)
        y = np.asarray(true, dtype=float).reshape(-1)
        if q.ndim != 2 or len(q) == 0:
            logging.warning("No quantile forecasts available for fan chart")
            return
        n = min(len(y), len(q))
        q, y = q[:n], y[:n]
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(x, q[:, 0], q[:, 6], color="#4C78A8", alpha=.16, label="P5-P95 (90%)")
        ax.fill_between(x, q[:, 1], q[:, 5], color="#4C78A8", alpha=.24, label="P10-P90 (80%)")
        ax.fill_between(x, q[:, 2], q[:, 4], color="#4C78A8", alpha=.38, label="P25-P75 (50%)")
        ax.plot(x, q[:, 3], color="#E45756", linewidth=2, label="P50")
        ax.plot(x, y, color="black", marker="o", linewidth=1.8, label="Actual")
        ax.set(title="Quantile Forecast Fan Chart", xlabel="Test month", ylabel="Quantity")
        ax.legend(ncol=3)
        fig.tight_layout()
        path = self.save_dir / f"scenario_{scenario}_fan_chart.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[OK] Saved: {path}")

    def plot_calibration_curve(self, quantiles, true, scenario: int = 1):
        q = np.asarray(quantiles, dtype=float)
        y = np.asarray(true, dtype=float).reshape(-1)
        if q.ndim != 2 or len(q) == 0:
            return
        levels = np.array([.05, .10, .25, .50, .75, .90, .95])
        n = min(len(y), len(q))
        empirical = np.array([np.mean(y[:n] <= q[:n, i]) for i in range(len(levels))])
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", label="Ideal calibration")
        ax.plot(levels, empirical, "o-", color="#E45756", linewidth=2, label="Empirical")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal quantile", ylabel="Empirical coverage",
               title="Quantile Calibration Curve")
        ax.legend()
        fig.tight_layout()
        path = self.save_dir / f"scenario_{scenario}_calibration_curve.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[OK] Saved: {path}")

    def plot_expected_cost_curve(self, quantiles, true, holding_cost: float,
                                 shortage_cost: float, scenario: int = 1):
        q = np.asarray(quantiles, dtype=float)
        d = np.asarray(true, dtype=float).reshape(-1)
        if q.ndim != 2 or len(q) == 0:
            return
        levels = np.array([.05, .10, .25, .50, .75, .90, .95])
        cr = shortage_cost / (shortage_cost + holding_cost)
        y_star = np.mean([np.interp(cr, levels, row) for row in q])
        lo, hi = min(d.min(), q.min()), max(d.max(), q.max())
        stock = np.linspace(max(0, lo * .8), hi * 1.1, 250)
        costs = np.array([
            holding_cost * np.mean(np.maximum(level - d, 0))
            + shortage_cost * np.mean(np.maximum(d - level, 0))
            for level in stock
        ])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock, costs, color="#4C78A8", linewidth=2)
        ax.axvline(y_star, color="#E45756", linestyle="--", label=f"CR={cr:.2f}, y*={y_star:,.0f}")
        ax.scatter([stock[np.argmin(costs)]], [costs.min()], color="black", zorder=3, label="Empirical minimum")
        ax.set(title="Expected Overage/Underage Cost Curve", xlabel="Prepared quantity y", ylabel="Expected variable cost")
        ax.legend()
        fig.tight_layout()
        path = self.save_dir / f"scenario_{scenario}_expected_cost_curve.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[OK] Saved: {path}")

    def plot_eda(self):
        """Create STL decomposition and annual exogenous distribution-shift charts."""
        import pandas as pd
        from statsmodels.tsa.seasonal import STL
        from src.data.data_loader import data_loader

        df = data_loader().copy()
        df["Month"] = pd.to_datetime(df["Month"])
        df = df.sort_values("Month")
        stl = STL(df["Quantity"].astype(float), period=12, robust=True).fit()
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for ax, values, label in zip(
            axes, [df["Quantity"], stl.trend, stl.seasonal, stl.resid],
            ["Observed", "Trend", "Seasonality", "Residual"],
        ):
            ax.plot(df["Month"], values, linewidth=1.5)
            ax.set_ylabel(label)
        axes[0].set_title("STL Decomposition of Quantity")
        fig.tight_layout()
        path = self.save_dir / "eda_stl_decomposition.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        df["Year"] = df["Month"].dt.year
        features = ["PromotionAmount", "CompetitorQuantity", "Imports", "IPI"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for ax, feature in zip(axes.ravel(), features):
            annual = df.groupby("Year")[feature].mean()
            normalized = (annual - annual.mean()) / (annual.std() + 1e-8)
            ax.plot(normalized.index, normalized.values, "o-", linewidth=2)
            ax.set_title(f"Annual mean shift: {feature}")
            ax.set_ylabel("Standardized annual mean")
        fig.tight_layout()
        path = self.save_dir / "eda_distribution_shift.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info("[OK] Saved EDA charts")

    def update_model_comparison(self, model_name: str, metrics: Dict[str, float], scenario: int = 1):
        """Persist the latest model result and plot TC/fill-rate comparisons."""
        import pandas as pd
        csv_path = self.save_dir / f"scenario_{scenario}_model_metrics.csv"
        row = pd.DataFrame([{
            "model": model_name.upper(),
            "Total_Cost": metrics.get("Total_Cost", metrics.get("TC_min", np.nan)),
            "Fill_Rate": metrics.get("fill_rate", np.nan),
            "MAPE": metrics.get("MAPE", np.nan),
        }])
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            old = old[old["model"] != model_name.upper()]
            row = pd.concat([old, row], ignore_index=True)
        row.to_csv(csv_path, index=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(row["model"], row["Total_Cost"], color="#4C78A8")
        axes[0].set_title("Total Cost by Model")
        axes[0].set_ylabel("Total Cost (cost units)")
        axes[1].bar(row["model"], row["Fill_Rate"] * 100, color="#59A14F")
        axes[1].set_title("Fill Rate by Model")
        axes[1].set_ylabel("Fill Rate (%)")
        axes[1].set_ylim(0, 100)
        fig.tight_layout()
        path = self.save_dir / f"scenario_{scenario}_inventory_model_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"[OK] Saved: {path}")
    
    def plot_predictions_vs_actual(self, pred: np.ndarray, true: np.ndarray, 
                                   scenario: int = 1):
        """Plot predictions vs actual values."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Time series comparison
        time_steps = np.arange(len(pred))
        axes[0, 0].plot(time_steps, true, 'o-', label='Actual', linewidth=2, 
                       markersize=5, color='#2E86AB', alpha=0.8)
        axes[0, 0].plot(time_steps, pred, 's--', label='Predicted', linewidth=2, 
                       markersize=5, color='#F18F01', alpha=0.8)
        axes[0, 0].set_xlabel('Time Step', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Quantity (Units)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Predictions vs Actual Values', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot (Actual vs Predicted)
        axes[0, 1].scatter(true, pred, alpha=0.6, s=80, color='#A23B72', edgecolors='black', linewidth=0.5)
        # Add diagonal line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Quantity', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Predicted Quantity', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals (Error)
        residuals = true - pred
        axes[1, 0].bar(time_steps, residuals, color=['#2E86AB' if r >= 0 else '#C1121F' for r in residuals],
                       alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].set_xlabel('Time Step', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Prediction Errors', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
        axes[1, 1].set_xlabel('Residual Value', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / f'scenario_{scenario}_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()
    
    def plot_test_metrics(self, metrics: Dict[str, float], scenario: int = 1):
        """Plot test performance metrics as bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract relevant metrics
        metric_names = ['MAE', 'RMSE', 'MAPE', 'TC_min']
        metric_values = []
        colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C1121F']
        
        for name in metric_names:
            if name in metrics:
                metric_values.append(metrics[name])
        
        # Create bar chart
        bars = ax.bar(metric_names[:len(metric_values)], metric_values, 
                      color=colors_list[:len(metric_values)], alpha=0.8, 
                      edgecolor='black', linewidth=1.5, width=0.6)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Scenario {scenario} - Test Performance Metrics', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add custom y-axis labels for each metric
        metric_labels = ['MAE (units)', 'RMSE (units)', 'MAPE (%)', 'TC_min (cost units)']
        ax.set_xticklabels([metric_labels[i] for i in range(len(metric_values))], fontsize=11)
        
        plt.tight_layout()
        save_path = self.save_dir / f'scenario_{scenario}_test_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()
    
    def plot_comparison_with_baseline(self, pred: np.ndarray, true: np.ndarray,
                                      baseline_predictions: Optional[np.ndarray] = None,
                                      scenario: int = 1):
        """Plot predictions vs naive baseline (if baseline provided)."""
        if baseline_predictions is None:
            # Naive baseline: last value prediction
            baseline_predictions = np.roll(true, 1)
            baseline_name = "Naive (Last Value)"
        else:
            baseline_name = "Baseline"
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        time_steps = np.arange(len(pred))
        ax.plot(time_steps, true, 'o-', label='Actual', linewidth=2.5, 
               markersize=6, color='black', alpha=0.8)
        ax.plot(time_steps, pred, 's--', label='Our Model', linewidth=2.5, 
               markersize=6, color='#2E86AB', alpha=0.8)
        ax.plot(time_steps, baseline_predictions, '^--', label=baseline_name, linewidth=2.5, 
               markersize=6, color='#C1121F', alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Quantity (Units)', fontsize=12, fontweight='bold')
        ax.set_title(f'Scenario {scenario} - Model vs {baseline_name}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / f'scenario_{scenario}_model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()
    
    def plot_metrics_summary(self, metrics: Dict[str, float], epoch: Optional[int] = None, 
                           scenario: int = 1):
        """Print metrics summary as text plot."""
        # Create figure with text
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Title
        title = f"Scenario {scenario} - Test Performance Summary"
        if epoch:
            title += f" (Best at Epoch {epoch})"
        
        ax.text(0.5, 0.95, title, transform=ax.transAxes, 
               fontsize=14, fontweight='bold', ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Metrics text
        y_position = 0.85
        metrics_text = []
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['MAE', 'MSE', 'RMSE', 'TC_min']:
                    metrics_text.append(f"{key:20s} : {value:>15.2f}")
                else:
                    metrics_text.append(f"{key:20s} : {value:>15.4f}")
        
        text_str = '\n'.join(metrics_text)
        ax.text(0.1, y_position, text_str, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        save_path = self.save_dir / f'scenario_{scenario}_metrics_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()

    def plot_decomposition_diagnostics(
        self,
        component_forecasts: Dict[str, np.ndarray],
        true: np.ndarray,
        scenario: int = 1,
    ):
        """Plot decomposition-specific diagnostics for decomp models."""
        trend = np.asarray(component_forecasts.get("trend", []), dtype=float).flatten()
        seasonal = np.asarray(component_forecasts.get("seasonal", []), dtype=float).flatten()
        combined = np.asarray(component_forecasts.get("combined", []), dtype=float).flatten()
        true = np.asarray(true, dtype=float).flatten()

        n = min(len(trend), len(seasonal), len(combined), len(true))
        if n == 0:
            logging.warning("No decomposition components to visualize")
            return

        trend = trend[:n]
        seasonal = seasonal[:n]
        combined = combined[:n]
        true = true[:n]
        t = np.arange(n)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Component forecasts over time
        axes[0, 0].plot(t, true, color="black", linewidth=2, label="Actual")
        axes[0, 0].plot(t, combined, color="#2E86AB", linestyle="--", linewidth=2, label="Combined")
        axes[0, 0].plot(t, trend, color="#F18F01", alpha=0.9, label="Trend component")
        axes[0, 0].plot(t, seasonal, color="#A23B72", alpha=0.9, label="Seasonal/Residual component")
        axes[0, 0].set_title("Decomposition Components vs Actual", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Time Step", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Value", fontsize=11, fontweight="bold")
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Contribution ratio by absolute magnitude
        abs_trend = np.abs(trend)
        abs_seasonal = np.abs(seasonal)
        denom = abs_trend + abs_seasonal + 1e-8
        trend_ratio = abs_trend / denom
        seasonal_ratio = abs_seasonal / denom
        axes[0, 1].plot(t, trend_ratio, color="#F18F01", linewidth=2, label="Trend ratio")
        axes[0, 1].plot(t, seasonal_ratio, color="#A23B72", linewidth=2, label="Seasonal ratio")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title("Component Contribution Ratio", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Time Step", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Ratio", fontsize=11, fontweight="bold")
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        # Residual over time
        residual = true - combined
        axes[1, 0].axhline(0, color="black", linewidth=1)
        axes[1, 0].plot(t, residual, color="#C1121F", linewidth=1.8)
        axes[1, 0].set_title("Final Residual (Actual - Combined)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Time Step", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Residual", fontsize=11, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # Predicted vs residual scatter for bias/heteroscedasticity check
        axes[1, 1].scatter(combined, residual, alpha=0.65, s=28, color="#2E86AB", edgecolors="black", linewidth=0.3)
        axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1.5)
        axes[1, 1].set_title("Residual vs Prediction", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Prediction", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Residual", fontsize=11, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / f"scenario_{scenario}_decomposition_diagnostics.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()

    def plot_time_series_decomposition(
        self,
        component_forecasts: Dict[str, np.ndarray],
        scenario: int = 1,
        seq_length: int = 12,
    ):
        """Plot the raw decomposition of the input time series (Original, Trend, Seasonal, Residual)."""
        raw_target = component_forecasts.get("raw_target")
        raw_trend = component_forecasts.get("raw_trend")
        raw_seasonal = component_forecasts.get("raw_seasonal")
        raw_residual = component_forecasts.get("raw_residual")

        if any(x is None or len(x) == 0 for x in [raw_target, raw_trend, raw_seasonal, raw_residual]):
            logging.warning("Missing some raw decomposition components. Skipping raw decomposition plot.")
            return

        # Helper to reconstruct continuous series from overlapping windows
        def reconstruct(flat_arr):
            n_samples = len(flat_arr) // seq_length
            if n_samples == 0:
                return np.array([])
            reshaped = flat_arr.reshape(n_samples, seq_length)
            return np.concatenate([reshaped[0], reshaped[1:, -1]])

        target = reconstruct(raw_target)
        trend = reconstruct(raw_trend)
        seasonal = reconstruct(raw_seasonal)
        residual = reconstruct(raw_residual)

        n = min(len(target), len(trend), len(seasonal), len(residual))
        if n == 0:
            logging.warning("Reconstructed series are empty. Skipping plot.")
            return

        target, trend, seasonal, residual = target[:n], trend[:n], seasonal[:n], residual[:n]
        t = np.arange(n)

        # Compute statistics for residual
        std_val = float(np.std(residual))
        median_val = float(np.median(residual))

        # Create plot: 4 subplots vertically stacked
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # 1. Original
        axes[0].plot(t, target, color="#2B2D42", linewidth=2, label="Original")
        axes[0].set_title(f"Scenario {scenario} — Time Series Decomposition", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Original", fontsize=11, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")

        # 2. Trend
        axes[1].plot(t, trend, color="#D90429", linewidth=2, label="Trend")
        axes[1].set_ylabel("Trend", fontsize=11, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left")

        # 3. Seasonal
        axes[2].plot(t, seasonal, color="#008080", linewidth=2, label="Seasonal")
        axes[2].set_ylabel("Seasonal", fontsize=11, fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper left")

        # 4. Residual
        axes[3].plot(t, residual, color="#8D99AE", linewidth=1.5, label="Residual")
        axes[3].axhline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        axes[3].set_ylabel("Residual", fontsize=11, fontweight="bold")
        axes[3].set_xlabel("Time Step", fontsize=11, fontweight="bold")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper left")

        # Add stats text box to the residual plot
        stats_text = f"Residual Stats:\nStd Dev = {std_val:.4f}\nMedian = {median_val:.4f}"
        props = dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#FFEBAA", alpha=0.9)
        axes[3].text(
            0.02, 0.95,
            stats_text,
            transform=axes[3].transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            bbox=props
        )

        plt.tight_layout()
        save_path = self.save_dir / f"scenario_{scenario}_time_series_decomposition.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"[OK] Saved: {save_path}")
        plt.close()
