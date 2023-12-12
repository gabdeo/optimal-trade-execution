import plotly.graph_objects as go
import numpy as np


class Visualizer:
    def __init__(self, trader, v):
        """
        Initialize the Visualization with a trader instance and a fixed volume vector.

        :param trader: An instance of the Trader class.
        :param v: A fixed volume vector to be used in visualizations.
        """
        self.trader = trader
        self.v = v

    def plot_sum_q_vs_cost(self, q_range):
        """
        Plot the cost against the sum of q using a 2D plot.

        :param q_range: A range of values for q to generate the plot.
        """
        sum_q = np.sum(q_range, axis=1)
        costs = np.array([self.trader.cost(q, self.v) for q in q_range])

        fig = go.Figure(data=[go.Scatter(x=sum_q, y=costs, mode="lines+markers")])
        fig.update_layout(
            title="Sum of q vs Cost", xaxis_title="Sum of q", yaxis_title="Cost"
        )
        fig.show()

    def plot_3d_cost(self, q_range):
        """
        Plot the cost in 3D when q has two dimensions.

        :param q_range: A range of values for q to generate the plot.
        """
        # Ensure that q_range is appropriate for a 3D plot
        if q_range.shape[1] != 2:
            raise ValueError("Dimension of q must be 2 for a 3D plot.")

        q1 = q_range[:, 0]
        q2 = q_range[:, 1]
        costs = np.array([self.trader.cost(q, self.v) for q in q_range])

        fig = go.Figure(
            data=[go.Mesh3d(x=q1, y=q2, z=costs, color="blue", opacity=0.50)]
        )
        fig.update_layout(
            title="3D Plot of Cost",
            scene=dict(xaxis_title="q1", yaxis_title="q2", zaxis_title="Cost"),
        )
        fig.show()
