import numpy as np


class Trader:
    def __init__(self, alpha):
        """
        Initialize the Trader with specified parameters.

        :param alpha: Price sensitivity, quantifies the influence of the trade volume
                      on the asset price.
        """
        self.alpha = alpha

    def cost(self, q, v):
        """
        Calculate the cost of the trading strategy.

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
                min_term = min(s, t)
                sqrt_term = (q[s] * q[t]) / (v[s] * v[t])
                total_cost += q[s] * q[t] * (min_term + self.alpha**2 * sqrt_term)
        return total_cost

    def veccost(self, q, v):
        """
        Calculate the cost based on the trading strategy using NumPy for vectorized operations.

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
        min_terms = np.minimum(s_indices, t_indices)

        # Calculate the sqrt term using NumPy broadcasting
        sqrt_terms = (q[s_indices] * q[t_indices]) / (v[s_indices] * v[t_indices])

        # Calculate the total cost using NumPy operations for the entire matrix
        total_cost = np.sum(
            q[s_indices] * q[t_indices] * (min_terms + self.alpha**2 * sqrt_terms)
        )

        return total_cost


q = list(range(10))
v = [1, 2] * 5
alpha = 0.1
trader = Trader(alpha)
trader.cost(q, v)

trader.veccost(q, v)
