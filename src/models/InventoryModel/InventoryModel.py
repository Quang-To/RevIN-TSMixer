import numpy as np
from scipy.stats import norm


class InventoryModel:
    def __init__(self, shortage_cost: float, holding_cost: float = 2.0, lead_time: int = 2, ordering_cost: float = 50000.0):
        self.shortage_cost = shortage_cost
        self.holding_cost = holding_cost
        self.lead_time = lead_time
        self.ordering_cost = ordering_cost

    def total_cost(self, demand_forecast: np.ndarray, forecast_errors: np.ndarray) -> float:
        mu = np.mean(demand_forecast)
        if mu <= 0:
            raise ValueError("Mean demand must be positive")
        
        if forecast_errors is not None and len(forecast_errors) > 0:
            errors = np.asarray(forecast_errors).reshape(-1)
            sigma = np.std(errors, ddof=0)
            sigma = sigma + 1e-5
        else:
            sigma = 1e-5
        q_star = np.sqrt(2 * self.ordering_cost * mu / self.holding_cost)
        alpha = 1 - (self.holding_cost * q_star) / (self.shortage_cost * mu)

        z = norm.ppf(alpha)
        safety_stock = z * sigma * np.sqrt(self.lead_time)
        reorder_point = mu * self.lead_time + safety_stock
        loss_function = norm.pdf(z) - z * (1 - norm.cdf(z))
        expected_shortage = sigma * np.sqrt(self.lead_time) * loss_function

        # ---- 9. Cost components ----
        ordering_cost_total = self.ordering_cost * (mu / q_star)

        holding_cost_total = self.holding_cost * (
            q_star / 2 + max(0.0, safety_stock)
        )

        shortage_cost_total = (
            self.shortage_cost * expected_shortage * mu / q_star
        )

        # ---- 10. Total cost ----
        total = ordering_cost_total + holding_cost_total + shortage_cost_total

        return total