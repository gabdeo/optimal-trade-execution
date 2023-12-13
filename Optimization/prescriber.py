import numpy as np


class Prescriber:
    def __init__(self, mode, pred, quantity, horizon) -> np.array:
        self.pred = pred
        self.Q = quantity
        self.T = horizon

    def prescription(self, v, X, y):
        """
        Make a prescription on the quantities to purchase.
        equal: equal-weighted prediction
        average: sample-average method
        reg: supervised learning method with regression
        RF: supervised learning method with random forest
        XGB: supervised learning method with XGBoost
        """
        if self.pred == "equal":
            return np.full(shape=self.T, fill_value=self.Q / self.T)
        elif self.pred == "average":
            
        elif self.pred == "reg":
            pass
        elif self.pred == "RF":
            pass
        elif self.pred == "XGB":
            pass
