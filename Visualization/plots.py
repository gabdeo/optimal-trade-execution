import plotly.graph_objects as go
import numpy as np


class Visualizer:
    def __init__(self, trader):
        """
        Initialize the Visualization with a trader instance and a fixed volume vector.

        :param trader: An instance of the Trader class.
        """
        self.trader = trader

    def plot_sum_q_vs_cost(self, q_range, v_range):
        """
        Plot the cost against the sum of q using a 2D plot.

        :param q_range: A range of vectors for quantities q.
        :param v_range: A range of vectors for volumes v.
        """
        sum_q = np.sum(q_range, axis=1)
        costs = np.array([self.trader.cost(q, v) for (q, v) in zip(q_range, v_range)])

        fig = go.Figure(data=[go.Scatter(x=sum_q, y=costs, mode="lines+markers")])
        fig.update_layout(
            title="Sum of q vs Cost", xaxis_title="Sum of q", yaxis_title="Cost"
        )
        fig.show()

    def plot_surface_cost(self, q_range, v):
        """
        Compute costs for all combinations of q1 and q2 and plot a surface.

        :param q_range: A range of vectors for quantities q.
        :param v: A vector of volumes.
        """
        q1_range, q2_range = q_range[:, 0], q_range[:, 1]
        q1, q2 = np.meshgrid(q1_range, q2_range)
        costs = np.zeros_like(q1)

        for i in range(q1.shape[0]):
            for j in range(q1.shape[1]):
                q = np.array([q1[i, j], q2[i, j]])
                costs[i, j] = self.trader.cost(q, v)

        # Now use plotly to create the surface plot
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Surface(z=costs, x=q1_range, y=q2_range)])
        fig.update_layout(
            title="Cost Surface",
            xaxis_title="q1",
            yaxis_title="q1",
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=65, r=50, b=65, t=90),
        )
        fig.show()


if __name__ == "__main__":
    from Optimization.trading import Trader
    from Data.datagenerator import DataGenerator

    # Configuration for data generation
    config = {
        "T": 2,  # Number of time periods
        "volume": {"max": 100000, "min": 100000, "generation_type": "equal"},
        "quantity": {"max": 10000, "min": 0, "generation_type": "visu"},
    }

    # Instantiate the DataGenerator
    data_generator = DataGenerator(config, n_samples=100)

    # Generate quantities and volumes
    q, v, _ = data_generator.generate_data()

    # Trader parameters
    alpha = 0.01  # Price sensitivity

    # Instantiate the Trader object
    trader = Trader(alpha)

    # For visualization, create a range or grid of q values as needed
    # If T == 1, create a 1D array of q values; if T == 2, create a 2D grid of q values
    # ...

    # Instantiate the Visualization object
    visu = Visualizer(trader)

    # Call the appropriate plotting methods
    if config["T"] == 1:
        visu.plot_sum_q_vs_cost(q, v)
    elif config["T"] == 2:
        visu.plot_surface_cost(q, v[0])
    else:
        print("Visualization for T > 2 is not implemented.")
