import torch
import torch.nn as nn
class TrendBranch(nn.Module):
    """
    Lightweight trend forecasting branch.
    Uses simple linear or shallow MLP to capture trend components.
    """
    def __init__(self, seq_length: int, pred_len: int, n_features: int = 1, 
                 hidden_dim: int = 32, n_layers: int = 1, dropout: float = 0.0):
        """
        Args:
            seq_length: Input sequence length
            pred_len: Prediction length
            n_features: Number of features
            hidden_dim: Hidden dimension for MLP
            n_layers: Number of layers (1 for linear, >1 for MLP)
            dropout: Dropout rate
        """
        super().__init__()
        self.seq_length = seq_length
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_layers = n_layers
        
        # Flatten input
        input_dim = seq_length * n_features
        
        if n_layers == 1:
            # Simple linear regression
            self.model = nn.Linear(input_dim, pred_len)
        else:
            # Shallow MLP
            layers = []
            in_dim = input_dim
            
            for i in range(n_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            
            layers.append(nn.Linear(in_dim, pred_len))
            self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, n_features)
            
        Returns:
            forecast: (batch, pred_len)
        """
        # Flatten: (batch, seq_length * n_features)
        x_flat = x.reshape(x.size(0), -1)
        
        # Forecast
        forecast = self.model(x_flat)
        
        return forecast


class SeasonalityBranch(nn.Module):
    """
    Seasonality/Residual forecasting branch.
    Wrapper that uses TSMixer or N-HiTS for seasonal component.
    """
    def __init__(self, seq_length: int, pred_len: int, n_features: int = 1,
                 model_type: str = "tsmixer", **model_kwargs):
        """
        Args:
            seq_length: Input sequence length
            pred_len: Prediction length
            n_features: Number of features
            model_type: "tsmixer" or "nhits"
            **model_kwargs: Additional arguments for the model
        """
        super().__init__()
        self.seq_length = seq_length
        self.pred_len = pred_len
        self.n_features = n_features
        self.model_type = model_type
        
        if model_type == "tsmixer":
            from src.models.ForecastModel.ForecastModel import ForecastModel
            self.model = ForecastModel(
                seq_length=seq_length,
                pred_len=pred_len,
                ff_dim=model_kwargs.get("ff_dim", 64),
                dropout=model_kwargs.get("dropout", 0.1),
                n_block=model_kwargs.get("n_block", 2)
            )
        elif model_type == "nhits":
            from src.models.NHITSModel.NHITSModel import NHITSModel
            self.model = NHITSModel(
                seq_length=seq_length,
                pred_len=pred_len,
                n_features=n_features,
                n_stacks=model_kwargs.get("n_stacks", 2),
                n_blocks=model_kwargs.get("n_blocks", 1),
                n_layers=model_kwargs.get("n_layers", 2),
                hidden_dim=model_kwargs.get("hidden_dim", 64),
                dropout=model_kwargs.get("dropout", 0.1)
            )
        elif model_type == "nbeats":
            from src.models.NBEATSModel.NBEATSModel import NBEATSModel
            self.model = NBEATSModel(
                seq_length=seq_length,
                pred_len=pred_len,
                n_features=n_features,
                n_stacks=model_kwargs.get("n_stacks", 2),
                n_layers=model_kwargs.get("n_layers", 2),
                layer_dim=model_kwargs.get("layer_dim", 64),
                dropout=model_kwargs.get("dropout", 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, n_features)
            
        Returns:
            forecast: (batch, pred_len)
        """
        return self.model(x)


class AggregationLayer(nn.Module):
    def __init__(self, pred_len: int, aggregation_method: str = "sum", 
                 learnable_weights: bool = False):
        """
        Args:
            pred_len: Prediction length
            aggregation_method: "sum", "weighted", or "attention"
            learnable_weights: Whether to learn aggregation weights
        """
        super().__init__()
        self.pred_len = pred_len
        self.aggregation_method = aggregation_method
        
        if aggregation_method == "weighted" and learnable_weights:
            self.trend_weight = nn.Parameter(torch.tensor(0.5))
            self.seasonal_weight = nn.Parameter(torch.tensor(0.5))
        elif aggregation_method == "attention":
            # Learn which component to emphasize at each time step
            self.attention = nn.Sequential(
                nn.Linear(pred_len * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, trend_forecast: torch.Tensor, seasonal_forecast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trend_forecast: (batch, pred_len)
            seasonal_forecast: (batch, pred_len)
            
        Returns:
            combined_forecast: (batch, pred_len)
        """
        if self.aggregation_method == "sum":
            return trend_forecast + seasonal_forecast
        
        elif self.aggregation_method == "weighted":
            if hasattr(self, 'trend_weight'):
                # Normalize weights
                total_weight = self.trend_weight + self.seasonal_weight
                w1 = self.trend_weight / (total_weight + 1e-8)
                w2 = self.seasonal_weight / (total_weight + 1e-8)
                return w1 * trend_forecast + w2 * seasonal_forecast
            else:
                return 0.5 * trend_forecast + 0.5 * seasonal_forecast
        
        elif self.aggregation_method == "attention":
            # Compute attention weights
            combined = torch.cat([trend_forecast, seasonal_forecast], dim=-1)
            weights = self.attention(combined)  # (batch, 2)
            return weights[:, 0:1] * trend_forecast + weights[:, 1:2] * seasonal_forecast
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
