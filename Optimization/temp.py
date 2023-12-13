import cvxpy as cp
from Optimization.trading import Trader


class Optimizer:
    def __init__(self):
        self.problem = None
        self.vars = None

    def optimize(self, T, Q, objective_function):
        """
        Sets up the optimization problem and run it.
        :param T: Time horizon (number of variables).
        :param Q: Total quantity to be optimized.
        """
        # Define variables - non-negative and continuous
        self.vars = cp.Variable(T, nonneg=True)

        # Define the constraint - the sum of all variables should be equal to Q
        constraints = [cp.sum(self.vars) == Q]

        # Define the objective
        objective = cp.Minimize(objective_function(self.vars))
        self.problem = cp.Problem(objective, constraints)
        self.problem.solve()

    def get_results(self):
        """
        Returns the optimization results.
        :return: Optimal values of variables and the optimal objective value.
        """
        optimal_vars = self.vars.value
        optimal_value = self.problem.value
        return optimal_vars, optimal_value

    def summary(self):
        """
        Prints a summary of the optimization results.
        """
        optimal_vars, optimal_value = self.get_results()
        print("Optimal Value:", optimal_value)
        print("Optimal Variable Values:", optimal_vars)


if __name__ == "__main__":
    T = 10
    Q = 100000
    v = [1] * T
    trader = Trader(alpha=0.1, sigma=1.0)
    f = lambda x: trader.model_veccost(x, v)
    optimizer = Optimizer()
    optimizer.optimize(T, Q, f)
