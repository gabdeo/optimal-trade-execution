# test.py
import numpy as np
import pytest
from Optimization.trading import Trader


def test_cost_functions():
    # Test parameters
    alpha = 0.01
    q = np.array([10, 20, 30, 40, 50])  # Example quantities
    v = np.array([1000, 2000, 3000, 4000, 5000])  # Example volumes

    # Initialize the Trader instance
    trader = Trader(alpha)

    # Calculate costs using both methods
    cost = trader.model_cost(q, v)
    veccost = trader.model_veccost(q, v)

    # Assert that both methods yield the same result infinitesimal tolerance
    np.testing.assert_almost_equal(cost, veccost, decimal=10)

    # Calculate simpled costs using both methods
    cost = trader.model_simple_cost(q, v)
    veccost = trader.model_simple_veccost(q, v)

    # Assert that both methods yield the same result infinitesimal tolerance
    np.testing.assert_almost_equal(cost, veccost, decimal=10)
