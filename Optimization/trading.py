import numpy as np
from scipy.optimize import minimize


class Trader:
    def __init__(self, alpha, sigma):
        """
        Initialize the Trader with specified parameters.

        :param alpha: Price sensitivity, quantifies the influence of the trade volume
                      on the asset price.
        :param sigma: Price volatility
        """
        self.alpha = alpha
        self.sigma = sigma

    def model_cost(self, q, v):
        """
        Calculate the modelized cost of the trading strategy.

        :param q: List or array of quantities q_t to buy at times t.
        :param v: List or array of volumes v_t traded at times t.
        :return: Total trading cost for the buying of the asset.
        """
        if len(q) != len(v):
            raise ValueError("The lengths of quantities and volumes must be equal.")

        T = len(q)
        total_cost = 0
        for t in range(T):
            for s in range(T):
                min_term = min(s, t) + 1
                sqrt_term = ((q[s] * q[t]) / (v[s] * v[t])) ** (1 / 2)
                total_cost += q[s] * q[t] * (min_term + self.alpha**2 * sqrt_term)
        return total_cost

    def model_veccost(self, q, v):
        """
        Calculate the modelized cost based on the trading strategy using NumPy for vectorized operations.

        :param q: List of NumPy array of quantities q_t to buy at times t.
        :param v: List or NumPy array of volumes v_t traded at times t.
        :return: Cost as per the defined objective function.
        """
        if len(q) != len(v):
            raise ValueError("The lengths of quantities and volumes must be equal.")

        if isinstance(q, list):
            q = np.array(q)
        if isinstance(v, list):
            v = np.array(v)
        T = len(q)
        # Create a 2D grid of t and s indices
        t_indices, s_indices = np.meshgrid(range(T), range(T), indexing="ij")

        # Calculate the min term using NumPy minimum for element-wise comparison
        min_terms = np.minimum(s_indices, t_indices) + 1

        # Calculate the sqrt term using NumPy broadcasting
        sqrt_terms = (
            (q[s_indices] * q[t_indices]) / (v[s_indices] * v[t_indices])
        ) ** (1 / 2)

        # Calculate the total cost using NumPy operations for the entire matrix
        total_cost = np.sum(
            q[s_indices] * q[t_indices] * (min_terms + self.alpha**2 * sqrt_terms)
        )

        return total_cost

    def model_simple_cost(self, q, v):
        """
        Calculate a simplified version of the cost of the trading strategy.

        :param q: List or array of quantities q_t to buy at times t.
        :param v: List or array of volumes v_t traded at times t.
        :return: Total trading cost for the buying of the asset.
        """
        if len(q) != len(v):
            raise ValueError("The lengths of quantities and volumes must be equal.")

        T = len(q)
        total_cost = 0
        for t in range(T):
            sum_term = self.alpha**2 * q[t] ** 3 / v[t] + (t + 1) * q[t] ** 2
            total_cost += self.sigma**2 * sum_term
        return total_cost

    def real_cost(self, q, p):
        """
        Calculate the real cost of the trading strategy.

        :param q: List or array of quantities q_t to buy at times t.
        :param p: List or array of prices p_t at times t.
        :return: Total trading cost for the buying of the asset.
        """
        if isinstance(q, list):
            q = np.array(q)
        if isinstance(p, list):
            p = np.array(p)

        return (p[0] - np.sum(p * q) / np.sum(q)) ** 2
