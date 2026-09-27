import numpy as np
from typing import Sequence
from scipy.stats import norm as _norm

DEFAULT_QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

class InventoryModel:
    def __init__(self, holding_cost: float = 4.0, shortage_cost: float = 30.0, ordering_cost: float = 50_000.0, lead_time: int = 1, quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,):
        self.holding_cost   = float(holding_cost)
        self.shortage_cost  = float(shortage_cost)
        self.ordering_cost  = float(ordering_cost)
        self.lead_time      = int(lead_time)
        self.quantile_levels = np.asarray(sorted(quantile_levels), dtype=np.float64)

    @property
    def critical_ratio(self) -> float:
        return self.shortage_cost / (self.shortage_cost + self.holding_cost)

    def quantile_order_level(self, quantile_forecasts: np.ndarray,) -> np.ndarray:
        CR = self.critical_ratio
        q  = np.asarray(quantile_forecasts, dtype=np.float64)

        if q.ndim == 1:
            return float(np.interp(CR, self.quantile_levels, q))
        else:
            return np.array([
                np.interp(CR, self.quantile_levels, row)
                for row in q
            ])

    def total_cost(self, demand_actual: np.ndarray, quantile_forecasts: np.ndarray,) -> tuple[float, dict]:
        d   = np.asarray(demand_actual,      dtype=np.float64).flatten()
        qf  = np.asarray(quantile_forecasts, dtype=np.float64)

        if qf.ndim == 3:
            qf = qf.reshape(-1, qf.shape[-1])
        if qf.ndim == 1:
            qf = qf[np.newaxis, :]          

        n = min(len(d), len(qf))
        d, qf = d[:n], qf[:n]

        CR = self.critical_ratio

        if n == 0:
            raise ValueError("demand_actual and quantile_forecasts must not be empty")
        if qf.shape[-1] != len(self.quantile_levels):
            raise ValueError(
                f"Expected {len(self.quantile_levels)} quantiles, got {qf.shape[-1]}"
            )
        p50_idx = int(np.argmin(np.abs(self.quantile_levels - 0.50)))
        p50 = qf[:, p50_idx]
        y_cr = np.clip(self.quantile_order_level(qf), 0.0, None)
        quantile_uplift = y_cr - p50
        quantile_uplift_mean = float(np.mean(quantile_uplift))

        holding_per_period  = np.maximum(y_cr - d, 0.0)   
        shortage_per_period = np.maximum(d  - y_cr, 0.0)  

        holding_units_total = float(np.sum(holding_per_period))
        shortage_units_total = float(np.sum(shortage_per_period))
        holding_total = self.holding_cost * holding_units_total
        shortage_total = self.shortage_cost * shortage_units_total

        mu = np.mean(np.clip(d, 1.0, None))
        annual_demand = 12.0 * mu
        q_star = np.sqrt(2.0 * self.ordering_cost * annual_demand / max(self.holding_cost, 1e-8))
        ordering_total = self.ordering_cost * (annual_demand / max(q_star, 1e-8))

        total = holding_total + shortage_total + ordering_total

        units_short     = np.maximum(d - y_cr, 0.0)
        fill_rate       = float(1.0 - np.sum(units_short) / max(np.sum(d), 1e-8))
        fill_rate       = float(np.clip(fill_rate, 0.0, 1.0))
        cycle_service_level = float(np.mean(d <= y_cr))
        stockout_months = int(np.sum(d > y_cr))              
        p50_forecast    = float(np.mean(qf[:, p50_idx]))

        components = {
            "holding_cost_total":  float(holding_total),
            "shortage_cost_total": float(shortage_total),
            "ordering_cost_total": float(ordering_total),
            "ordering_cost_in_realized_tc": True,
            "cost_unit":             "configured_cost_unit",
            "holding_units_total":   holding_units_total,
            "shortage_units_total":  shortage_units_total,
            # Inventory parameters
            "critical_ratio":      float(CR),
            "y_cr_mean":           float(np.mean(y_cr)),
            "q_star":              float(q_star),
            "mu":                  float(mu),
            "annual_demand":       float(annual_demand),
            "quantile_uplift_mean": quantile_uplift_mean,
            # Note: y_cr_mean above is the newsvendor prep level yCR, NOT the
            # true (r,q) reorder point r = mu*L + SS (see total_cost_point).
            # 3-tier metrics
            "fill_rate":           fill_rate,
            "cycle_service_level": cycle_service_level,
            "stockout_months":     stockout_months,
            "p50_forecast_mean":   p50_forecast,
        }
        return float(total) if np.isfinite(total) else float("inf"), components


    def total_cost_point(self, demand_forecast: np.ndarray, forecast_errors: np.ndarray,) -> tuple[float, dict]:
        mu    = float(np.mean(np.clip(demand_forecast, 1.0, None)))
        annual_demand = 12.0 * mu
        sigma = float(np.std(forecast_errors, ddof=1)) + 1e-5 if len(forecast_errors) > 1 else 1e-5
        CR    = self.critical_ratio
        z     = _norm.ppf(CR)
        safety_stock   = z * sigma * np.sqrt(self.lead_time)
        q_star         = np.sqrt(2.0 * self.ordering_cost * annual_demand / max(self.holding_cost, 1e-8))
        reorder_point  = mu * self.lead_time + safety_stock
        loss_fn        = _norm.pdf(z) - z * (1 - _norm.cdf(z))
        expected_short = sigma * np.sqrt(self.lead_time) * loss_fn
        ordering_total = self.ordering_cost * (annual_demand / max(q_star, 1e-8))
        holding_total = self.holding_cost * (q_star / 2 + max(0.0, safety_stock))
        shortage_total = self.shortage_cost * expected_short * annual_demand / max(q_star, 1e-8)
        total          = ordering_total + holding_total + shortage_total
        return float(total), {
            "ordering_cost_total": float(ordering_total),
            "holding_cost_total":  float(holding_total),
            "shortage_cost_total": float(shortage_total),
            "q_star":              float(q_star),
            "safety_stock":        float(safety_stock),
            "reorder_point":       float(reorder_point),
            "mu":                  float(mu),
            "annual_demand":       float(annual_demand),
            "sigma":               float(sigma),
            "critical_ratio":      float(CR),
            "cost_unit":           "configured_cost_unit",
        }