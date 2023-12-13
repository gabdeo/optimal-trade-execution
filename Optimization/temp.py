import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Optimization.trading import Trader
from tqdm import tqdm

df = pd.read_csv(
    "Data\predictions\preds.csv",
    index_col=[0, 1],
    parse_dates=["Date"],
    date_format="%Y-%m-%d %H:%M:%S",
)

from Optimization.optimizer import Optimizer

# dates = df["Date"].unique()
# tickers = df["Ticker"].unique()
targets = list(df.columns)[:2]
T = 7
Q = 50000 * T
trader = Trader(alpha=0.1, sigma=1)
results = pd.DataFrame(columns=[f"Prescip_{k}" for k in range(1, 8)], index=df.index)

# for k in range(1, 8):
#     df[f"Prescip_{k}"] = 0.0
for  idx, row in tqdm(df.iterrows(), total=len(df)):
    # ticker = row[0]
    # date = row[1]
    log_volumes = np.array(row, dtype=float)
    f = lambda x: trader.model_veccost(x, np.exp(log_volumes))
    optimizer = Optimizer()
    result = optimizer.optimize(Q, T, f)
    # print("opti done")
    results.loc[idx, :] = result.x
    # for k in range(1, 8):
    #     df[f"Prescip_{k}"][i] = result.x[k - 1]

df_prices = pd.read_csv('Data/tables/snp500.csv',
                        parse_dates=['Date'], 
                        index_col=[0, 1])
prices = df_prices[['Open', 'Volume']]
results = pd.read_csv('reg_prescriptions.csv', parse_dates=['Date'], index_col=[0, 1])

# Extracting date and time components
new_index = [(ticker, datetime.date(), datetime.time()) for ticker, datetime in prices.index]

# Creating a new multi-index
prices.index = pd.MultiIndex.from_tuples(new_index, names=['Ticker', 'Date', 'time'])
prices = prices.groupby(['Ticker', 'Date']).agg(list)
prices = prices[prices['Open'].apply(len) == 7]

# Create new DataFrames for each set of columns
df_open = pd.DataFrame(prices['Open'].tolist(), columns=[f'Open_{i+1}' for i in range(7)], index=prices.index)
df_volume = pd.DataFrame(prices['Volume'].tolist(), columns=[f'Volume_{i+1}' for i in range(7)], index=prices.index)

# Concatenate the new DataFrames horizontally
prices_volumes = pd.concat([df_open, df_volume], axis=1)
prices_volumes.index.names = results.index.names

df_merge = pd.merge(results, prices_volumes, left_index=True, right_index=True, how='inner')
for i in range(7):
    df_merge = df_merge[df_merge[f'Volume_{i+1}'] != 0.]

prices = df_merge[[f'Open_{i+1}' for i in range(7)]]

volumes = df_merge[[f'Volume_{i+1}' for i in range(7)]]

quantities = df_merge[[f'Prescip_{i+1}' for i in range(7)]]

trader = Trader(alpha=.1,sigma=1)
costs = trader.real_cost(quantities.values, prices.values, volumes.values)

shape = quantities.shape
benchmark_quantities = np.full(shape, fill_value=Q/shape[1])
benchmark_costs = trader.real_cost(benchmark_quantities, prices.values, volumes.values)
df_costs = pd.concat([pd.DataFrame(costs, columns=['Strategy Cost'], index=prices.index), pd.DataFrame(costs, columns=['Benchmark Cost'], index=prices.index)] 
