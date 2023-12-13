import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def gaussian_random_walk(n, p_0, mu, sigma):
    """
    Generates a Gaussian random walk.

    :param n: Number of iterations
    :param p_0: Starting point
    :param mu: Mean of the Gaussian distribution
    :param sigma: Standard deviation of the Gaussian distribution
    :return: A numpy array containing the random walk
    """
    # Gaussian random steps
    steps = np.random.normal(mu, sigma, n)

    # Cumulative sum to get the position at each step
    walk = np.cumsum(steps)

    # Adding the starting position
    walk = p_0 + walk

    return walk


def price_impact(p, q, alpha):
    return p + alpha * np.cumsum(np.sqrt(q))


# Parameters
n = 25  # number of iterations
p_0 = 0  # starting point
mu = 0.4  # mean
sigma = 1  # standard deviation
quantity = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 7, 0, 0, 0, 0]
)
quantity = quantity / np.sum(quantity)
alpha = 1.0


# Generate the random walk
walk = gaussian_random_walk(n, p_0, mu, sigma)

# Generate Price Impact
price = price_impact(walk, quantity, alpha)

# Creating a subplot with 2 y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Adding the Random Walk line plot
fig.add_trace(
    go.Scatter(
        x=list(range(n)),
        y=walk,
        mode="lines",
        name="Random Walk",
        line=dict(color="green"),
    ),
    secondary_y=False,
)

# Adding the Elastic Price line plot
fig.add_trace(
    go.Scatter(
        x=list(range(n)),
        y=price,
        mode="lines",
        name="Elastic Price",
        line=dict(color="red"),
    ),
    secondary_y=False,
)

# Adding the Quantity bar plot with bars touching each other
fig.add_trace(
    go.Bar(
        x=list(range(n)),
        y=30 * np.cumsum(quantity),
        name="Quantity",
        marker_color="blue",
        opacity=0.5,
        width=1,
    ),
    secondary_y=True,
)

# Setting the axis titles
fig.update_xaxes(title_text="Steps")
fig.update_yaxes(title_text="Position", secondary_y=False)
fig.update_yaxes(title_text="Purchase (%)", range=[0, 100], secondary_y=True)

# Adding the plot title and grid lines
fig.update_layout(
    title="Purchase Execution Strategy", showlegend=True, plot_bgcolor="white"
)

# Showing the plot
fig.show()

# # Assuming 'quantity' is an array with the same length as 'walk' and 'price'
# # Adjust the following line if this is not the case
# x_coords = range(len(quantity))

# # Plotting
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Line plots on the primary y-axis
# ax1.plot(walk, label='Random Walk', color='green')
# ax1.plot(price, label='Elastic Price', color='red')

# # Adding labels, title, and legend for the primary y-axis
# ax1.set_xlabel("Steps")
# ax1.set_ylabel("Position", color='black')
# ax1.tick_params(axis='y', labelcolor='black')
# ax1.legend(loc='upper left')

# # Create a second y-axis for the bar plot
# ax2 = ax1.twinx()

# # Bar plot on the secondary y-axis
# ax2.bar(range(n), 30 * np.cumsum(quantity), alpha=0.5, label='Quantity', color='blue', width=1.0)

# # Adding labels and legend for the secondary y-axis
# ax2.set_ylabel("Purchase (%)", color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')
# ax2.set_ylim(0, 100)  # Setting the limit for quantity axis
# ax2.legend(loc='upper right')

# # Set the grid
# ax1.grid(False)

# # Show the plot
# plt.title("Purchase Execution Strategy")
# plt.show()
