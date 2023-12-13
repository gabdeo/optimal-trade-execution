import numpy as np
from scipy.optimize import minimize

from Optimization.trading import Trader


class Optimizer:
    def __init__(self):
        self.vars = {}

    def optimize(self, Q, T, objective):
        """
        Sets up the optimization problem by adding variables, constraints, and defining the objective function.
        :param variables: List of tuples in the form (name, lb, ub, var_type).
        :param constraints: List of tuples in the form (expr, sense, rhs).
        :param objective: Linear expression for the objective function.
        :param sense: Objective sense (e.g., GRB.MAXIMIZE, GRB.MINIMIZE).
        """

        # Add constraints
        # T = len(v)
        constraint = {"type": "eq", "fun": lambda q: np.sum(q) - Q}
        bounds = [(0, None) for _ in range(T)]
        initial_guess = np.ones(T) * (Q / T)

        result = minimize(
            objective, initial_guess, bounds=bounds, constraints=constraint
        )
        return result

    def get_results(self):
        """
        Returns the optimization results.
        :return: Tuple containing optimal values of variables, the optimal objective value, and the model status.
        """
        optimal_vars = {v.varName: v.x for v in self.model.getVars()}
        optimal_value = self.model.objVal
        status = self.model.Status
        return optimal_vars, optimal_value, status

    def summary(self):
        """
        Prints a summary of the optimization results.
        """
        optimal_vars, optimal_value, status = self.get_results()
        print("Optimization Status:", status)
        print("Optimal Value:", optimal_value)
        print("Optimal Variable Values:")
        for var_name, value in optimal_vars.items():
            print(f"{var_name}: {value}")


if __name__ == "__main__":
    T = 10
    Q = 100000
    v = [1] * T
    trader = Trader(alpha=0.1, sigma=1)
    f = lambda x: trader.model_veccost(x, v)

    optimizer = Optimizer()
    result = optimizer.optimize(Q, T, f)


# Example usage:
# optimizer = Optimizer()
# optimizer.setup(
#     variables=[...],
#     constraints=[...],
#     objective=...,
#     sense=GRB.MAXIMIZE
# )
# optimizer.optimize()
# optimizer.print_summary()
