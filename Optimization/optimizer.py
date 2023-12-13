from gurobipy import Model, GRB, quicksum
from scipy.optimize import minimize

from Optimization.trading import Trader


class Optimizer:
    def __init__(self):
        self.model = Model("Optimization Model")
        self.vars = {}

    def setup(self, variables, constraints, objective, sense):
        """
        Sets up the optimization problem by adding variables, constraints, and defining the objective function.
        :param variables: List of tuples in the form (name, lb, ub, var_type).
        :param constraints: List of tuples in the form (expr, sense, rhs).
        :param objective: Linear expression for the objective function.
        :param sense: Objective sense (e.g., GRB.MAXIMIZE, GRB.MINIMIZE).
        """
        # Add variables
        self.vars = {}
        for var_name, lb, ub, var_type in variables:
            self.vars[var_name] = self.model.addVar(
                lb=lb, ub=ub, vtype=var_type, name=var_name
            )

        self.model.update()

        # Add constraints
        for expr, sense, rhs in constraints:
            self.model.addConstr(expr, sense, rhs)

        # Set objective
        self.model.setObjective(objective, sense)

    def optimize(self):
        """
        Runs the optimization.
        """
        self.model.optimize()

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

    def optimize_function(self, f, v, T):
        """
        Optimizes the function `f`.
        :param f: The function to be optimized.
        :param trader: An instance of the Trader class.
        :param T: Time horizon.
        """
        # Define variables - non-negative and continuous
        variables = [
            ("q{}".format(t), 0, GRB.INFINITY, GRB.CONTINUOUS) for t in range(T)
        ]

        # Adding variables to the model
        for var_name, lb, ub, var_type in variables:
            self.vars[var_name] = self.model.addVar(
                lb=lb, ub=ub, vtype=var_type, name=var_name
            )

        # Update the model to integrate new variables
        self.model.update()

        # Define the constraint - the sum of all q variables should be equal to Q
        constraint_expr = quicksum(self.vars["q{}".format(t)] for t in range(T))
        self.model.addConstr(constraint_expr, GRB.EQUAL, Q)

        # Define the objective function using `f`
        objective = f([self.vars["q{}".format(t)] for t in range(T)])
        self.model.setObjective(objective, GRB.MINIMIZE)

        self.optimize()


if __name__ == "__main__":
    T = 10
    Q = 100000
    v = [1] * T
    trader = Trader(alpha=0.1, sigma=1)
    f = lambda x: trader.model_veccost(x, v)
    # Define variables - non-negative and continuous
    variables = [("q{}".format(t), 0, GRB.INFINITY, GRB.CONTINUOUS) for t in range(T)]

    # Define constraints
    # The sum of all q variables should be equal to Q
    # constraints = [(quicksum(q[t] for t in range(T)) == Q, GRB.EQUAL, Q)]
    
    optimizer = Optimizer()
    optimizer.optimize_function(f, trader, T)
    optimizer.summary()


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
