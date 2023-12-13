import numpy as np
import pandas as pd
from Optimization.trading import Trader

df = pd.read_csv("Data\predictions\preds.csv", index_col=[0,1], parse_dates=["Date"], date_format='%Y-%m-%d %H:%M:%S')

from Optimization.optimizer import Optimizer

# dates = df["Date"].unique()
# tickers = df["Ticker"].unique()
targets = list(df.columns)[:2]
T = 7
Q = 50000
trader = Trader(alpha=0.1, sigma=1)
results = pd.DataFrame(columns=[f"Prescip_{k}" for k in range(1, 8)], index = df.index)

# for k in range(1, 8):
#     df[f"Prescip_{k}"] = 0.0
for i, row in df.iterrows():
    # ticker = row[0]
    # date = row[1]
    log_volumes = np.array(row, dtype=float)
    f = lambda x: trader.model_veccost(x, np.exp(log_volumes))
    optimizer = Optimizer()
    result = optimizer.optimize(Q, T, f)
    # print("opti done")
    results.loc[i, :] = result.x
    # for k in range(1, 8):
    #     df[f"Prescip_{k}"][i] = result.x[k - 1]

pass
