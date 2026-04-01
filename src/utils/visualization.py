import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        metric_name = 'TC_min (đồng)' if scenario == 2 else 'MAPE (%)'
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
        print(f"✓ Saved: {save_path}")
        plt.close()
    
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
        print(f"✓ Saved: {save_path}")
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
        metric_labels = ['MAE (units)', 'RMSE (units)', 'MAPE (%)', 'TC_min (đồng)']
        ax.set_xticklabels([metric_labels[i] for i in range(len(metric_values))], fontsize=11)
        
        plt.tight_layout()
        save_path = self.save_dir / f'scenario_{scenario}_test_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
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
        print(f"✓ Saved: {save_path}")
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
        print(f"✓ Saved: {save_path}")
        plt.close()
