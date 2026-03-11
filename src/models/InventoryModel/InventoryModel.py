import numpy as np
from scipy.stats import norm

class InventoryModel:
    def __init__(self, shortage_cost: float, holding_cost: int = 2, lead_time: int = 2, ordering_cost: int = 50000):
        self.shortage_cost = shortage_cost
        self.holding_cost = holding_cost
        self.lead_time = lead_time
        self.ordering_cost = ordering_cost

    def total_cost(self, demand_forecast: np.ndarray) -> float:
        self.muy = np.mean(demand_forecast)
        self.sigma = np.std(demand_forecast)
        self.order_quantity = np.sqrt(2 * self.ordering_cost * self.muy / self.holding_cost)

        self.alpha = 1 - (self.holding_cost * self.order_quantity) / (self.shortage_cost * self.muy)
        self.z = norm.ppf(self.alpha)
        self.safety_stock = self.z * self.sigma * np.sqrt(self.lead_time)
        self.reorder_point = self.muy * self.lead_time + self.safety_stock
        self.loss_function = norm.pdf(self.z) - self.z * (1 - norm.cdf(self.z))
        self.expect_shortage = self.sigma * np.sqrt(self.lead_time) * self.loss_function
        self.cost_shortage = (self.shortage_cost * self.expect_shortage * self.muy) / self.order_quantity
        self.total_ordering_cost = self.ordering_cost * (self.muy / self.order_quantity)
        self.total_holding_cost = self.holding_cost * (self.order_quantity / 2 + max(0, self.safety_stock))
        self.total = self.total_ordering_cost + self.total_holding_cost + self.cost_shortage
        return self.total