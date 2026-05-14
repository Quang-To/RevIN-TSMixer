"""
Decomposed Forecast Model
Combines decomposition with separate trend and seasonality forecasting branches
"""
import torch
import torch.nn as nn
from src.models.Decomposition.TimeSeriesDecomposition import TimeSeriesDecomposition
from src.models.Decomposition.Branches import TrendBranch, SeasonalityBranch, AggregationLayer


class DecomposedForecastModel(nn.Module):
    """
    Time series forecasting with decomposition strategy.
    
    Architecture:
    1. Decompose input into trend, seasonal, and residual
    2. Trend branch: lightweight linear/MLP
    3. Seasonality branch: TSMixer or N-HiTS
    4. Aggregate predictions from both branches
    
    Motivation:
    - Reduces burden on main backbone by separating concerns
    - Trend is simpler (linear patterns) -> lightweight model
    - Seasonality is complex (cyclical patterns) -> complex model
    - Better fits with limited data by simplifying trend modeling
    """
    
    def __init__(
        self,
        seq_length: int,
        pred_len: int,
        n_features: int = 1,
        seasonal_period: int = 4,
        decomposition_method: str = "ma",
        # Trend branch
        trend_hidden_dim: int = 32,
        trend_n_layers: int = 1,
        # Seasonality branch
        seasonality_model: str = "tsmixer",  # "tsmixer", "nhits", or "nbeats"
        seasonality_hidden_dim: int = 64,
        seasonality_n_blocks: int = 2,
        seasonality_n_stacks: int = 2,
        seasonality_n_layers: int = 2,
        seasonality_layer_dim: int = 64,
        # Common
        dropout: float = 0.1,
        aggregation_method: str = "sum",
        learnable_aggregation: bool = False
    ):
        """
        Args:
            seq_length: Input sequence length
            pred_len: Prediction length
            n_features: Number of input features
            seasonal_period: Expected seasonal period
            decomposition_method: "ma" or "savgol"
            
            Trend branch parameters:
            - trend_hidden_dim: Hidden dimension for MLP
            - trend_n_layers: Number of layers (1 = linear)
            
            Seasonality branch parameters:
            - seasonality_model: Model type for seasonality
            - seasonality_hidden_dim: Hidden dimension
            - seasonality_n_blocks: Number of blocks
            
            Aggregation:
            - aggregation_method: "sum", "weighted", or "attention"
            - learnable_aggregation: Learn aggregation weights
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.pred_len = pred_len
        self.n_features = n_features
        self.decomposition_method = decomposition_method
        self.seasonality_model_type = seasonality_model
        
        # 1. Decomposition layer
        self.decomposition = TimeSeriesDecomposition(
            seq_length=seq_length,
            seasonal_period=seasonal_period,
            method=decomposition_method
        )
        
        # 2. Trend branch - lightweight
        self.trend_branch = TrendBranch(
            seq_length=seq_length,
            pred_len=pred_len,
            n_features=n_features,
            hidden_dim=trend_hidden_dim,
            n_layers=trend_n_layers,
            dropout=dropout
        )
        
        # 3. Seasonality branch - TSMixer/N-HiTS
        self.seasonality_branch = SeasonalityBranch(
            seq_length=seq_length,
            pred_len=pred_len,
            n_features=n_features,
            model_type=seasonality_model,
            hidden_dim=seasonality_hidden_dim,
            n_blocks=seasonality_n_blocks,
            n_stacks=seasonality_n_stacks,
            n_layers=seasonality_n_layers,
            layer_dim=seasonality_layer_dim,
            dropout=dropout
        )
        
        # 4. Aggregation layer
        self.aggregation = AggregationLayer(
            pred_len=pred_len,
            aggregation_method=aggregation_method,
            learnable_weights=learnable_aggregation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Decompose
        trend, seasonal, residual = self.decomposition(x)
        
        # Step 2: Combine seasonal and residual for seasonality branch
        # seasonal + residual captures the oscillatory and noise components
        seasonal_residual = seasonal + residual
        
        # Step 3: Forecast from each branch
        trend_forecast = self.trend_branch(trend)           # (batch, pred_len)
        seasonal_forecast = self.seasonality_branch(seasonal_residual)  # (batch, pred_len)
        
        # Step 4: Aggregate
        forecast = self.aggregation(trend_forecast, seasonal_forecast)  # (batch, pred_len)
        
        return forecast
    
    def get_component_forecasts(self, x: torch.Tensor) -> dict:
        """
        Get individual component forecasts for analysis.
        
        Args:
            x: (batch, seq_length, n_features)
            
        Returns:
            dict with keys: "trend", "seasonal", "combined"
        """
        trend, seasonal, residual = self.decomposition(x)
        seasonal_residual = seasonal + residual
        
        trend_forecast = self.trend_branch(trend)
        seasonal_forecast = self.seasonality_branch(seasonal_residual)
        combined = self.aggregation(trend_forecast, seasonal_forecast)
        
        return {
            "trend": trend_forecast,
            "seasonal": seasonal_forecast,
            "combined": combined
        }


class HierarchicalDecomposedModel(nn.Module):
    """
    Advanced version with hierarchical decomposition:
    - Level 1: Trend vs Fluctuation
    - Level 2: Seasonality vs Residual (from fluctuation)
    """
    
    def __init__(
        self,
        seq_length: int,
        pred_len: int,
        n_features: int = 1,
        seasonal_period: int = 4,
        dropout: float = 0.1,
        seasonality_model: str = "tsmixer"
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.pred_len = pred_len
        
        # Level 1 decomposition: Trend vs Fluctuation
        self.level1_decomposition = TimeSeriesDecomposition(
            seq_length=seq_length,
            seasonal_period=seasonal_period * 2,  # Larger window for trend
            method="ma"
        )
        
        # Level 2 decomposition: Seasonality vs Residual (from fluctuation)
        self.level2_decomposition = TimeSeriesDecomposition(
            seq_length=seq_length,
            seasonal_period=seasonal_period,
            method="ma"
        )
        
        # Three branches
        self.trend_branch = TrendBranch(
            seq_length=seq_length,
            pred_len=pred_len,
            n_features=n_features,
            hidden_dim=32,
            n_layers=1,
            dropout=dropout
        )
        
        self.seasonal_branch = SeasonalityBranch(
            seq_length=seq_length,
            pred_len=pred_len,
            n_features=n_features,
            model_type=seasonality_model,
            hidden_dim=64,
            dropout=dropout
        )
        
        self.residual_branch = SeasonalityBranch(
            seq_length=seq_length,
            pred_len=pred_len,
            n_features=n_features,
            model_type=seasonality_model,
            hidden_dim=48,
            dropout=dropout
        )
        
        # Aggregation
        self.aggregation = AggregationLayer(
            pred_len=pred_len,
            aggregation_method="weighted",
            learnable_weights=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, n_features)
            
        Returns:
            forecast: (batch, pred_len)
        """
        # Level 1: Trend vs Fluctuation
        trend, fluctuation, _ = self.level1_decomposition(x)
        
        # Level 2: Decompose fluctuation into seasonality and residual
        _, seasonal, residual = self.level2_decomposition(fluctuation)
        
        # Forecasts
        trend_pred = self.trend_branch(trend)
        seasonal_pred = self.seasonal_branch(seasonal)
        residual_pred = self.residual_branch(residual)
        
        # Simple aggregation (can be extended with attention)
        forecast = trend_pred + seasonal_pred + residual_pred
        
        return forecast
