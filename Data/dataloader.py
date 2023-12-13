import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from warnings import warn

class DataLoader():

    def __init__(self, tickers: list[str], start_date: str, end_date: str, interval="1h", date_fromat='%Y-%m-%d'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.date_format = date_fromat

    def load_data(self, local_path = None, metrics = None, preprocess = True):

        if local_path is not None:
            data = pd.read_csv(local_path, index_col=[0,1], parse_dates=["Date"], date_format='%Y-%m-%d %H:%M:%S')

            if metrics and not all([m in data.columns for m in metrics]):
                raise ValueError("Metrics are not in the loaded data")
            if metrics:
                data = data[metrics]
            

        else:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date, interval=self.interval, actions=True, ignore_tz=True)
            returns = data["Open"].pct_change()
            returns.columns =  pd.MultiIndex.from_tuples([('Returns', col) for col in returns.columns])
            data = pd.concat([data, returns], axis=1)
            data = data.stack(level=1) # Columns: features, Rows: date, ticker
            
            if preprocess:
                self.data = data
                data = self.preprocess_data()

            if metrics:
                data = data[metrics]

            data = data.swaplevel().sort_index() # Sort by ticker then by date
            data.index.names = ['Ticker', 'Date']
        
        self.data = data
        return data
    
    def save_data(self, path):
        self.data.to_csv(path)
    
    def preprocess_data(self, nan_threshold=0.8, verbose = False):

        if verbose:
            print("Number of NaN values per column:")
            print(self.data.isna().sum())

        self.data = self.data.dropna(axis=1, thresh=int(nan_threshold * self.data.shape[0]))
        self.data = self.data.ffill()

        if self.data.isna().sum().sum() > 0:
            warn("NaN values remain in the dataset.")
            self.data = self.data.dropna()

        return self.data
    
    def join_macro_data(self, tickers, interval = None):
        if interval is None:
            interval = self.interval
        macro_data = yf.download(tickers, start=self.start_date, end=self.end_date, interval=interval)["Open"]
        macro_data = pd.DataFrame(macro_data)
        macro_data.columns = tickers

        self.data = pd.merge_asof(self.data.reset_index().sort_values(by="Date"), macro_data.reset_index(), left_on = "Date", right_on="Date").set_index(['Ticker', 'Date'])
        self.data.index.names = ['Ticker', 'Date']
        # self.data = self.data.drop(columns=["Date"])
        self.data = self.data.sort_index()

        return self.data
    
    def onehot_encode(self, columns):
        
        to_concat = [self.data]
        for column in columns:
            prefix = column
            to_concat.append(pd.get_dummies(self.data[column], prefix=prefix).astype(int))

        self.data = pd.concat(to_concat, axis=1)
        self.data = self.data.drop(columns=columns)
        return self.data

    def rolling_apply(self, func, window, on_cols, new_names, drop_col=False, dropna = True, **kwargs):
        self.data[new_names] = self.data[on_cols].groupby(level="Ticker").rolling(window=window).apply(func, **kwargs).droplevel(0)
        if drop_col:
            self.data = self.data.drop(columns=on_cols)
        if dropna:
            self.data = self.data.dropna()

        return self.data
    
    def prepare_features(self, features, target, shift_features = True, daily_features=[]):
        
        aggregations = {col:'mean' if col in daily_features else lambda x: list(x) for col in features}

        
        data = self.data.copy()
        new_index = [(ticker, datetime.date(), datetime.time()) for ticker, datetime in data.index]
        data.index = pd.MultiIndex.from_tuples(new_index, names=['Ticker', 'Date', 'Time'])

        data = data.groupby(["Ticker", "Date"]).agg(aggregations)


        if shift_features:
            data["Target"] = data["Volume"].groupby("Ticker").shift(-1)
            data = data.dropna()
        else:
            data["Target"] = data["Volume"]
        
        # Flattens list aggregations
        for col in data.columns:
            if aggregations.get(col, None) == 'mean':
                continue
            list_length = max(data[col].apply(len))
            data = data[data[col].apply(len) == list_length]

            # Create new columns for each element in the list
            for i in range(list_length):
                new_col_name = f'{col}_{i+1}'
                data[new_col_name] = data[col].apply(lambda x: x[i])
            data = data.drop(columns=[col])

        self.data = data
        self.target = data[[col for col in data.columns if col.startswith("Target")]]
        self.features = data.drop(columns=[col for col in data.columns if col.startswith("Target")])

        return data


if __name__ == '__main__':
    
    with open('Data/tickers/snp500.txt', 'r') as f:
        tickers = f.read().splitlines()
    # tickers = tickers[:5]

    start_date = "2022-01-01"
    end_date = "2023-12-10"

    dataset = DataLoader(tickers, start_date, end_date)
    data = dataset.load_data()
    
    alt_data = pd.read_excel('Data/alt_features/industries_exchanges.xlsx', index_col=1)
    data = pd.merge(data.reset_index(), alt_data.reset_index(), how="left", left_on="Ticker", right_on='Symbol')
    data = data.set_index(['Ticker', 'Date'])
    data.index.names = ['Ticker', 'Date']
    dataset.data = data

    # Add VIX
    dataset.join_macro_data(["^VIX"], interval = "1d")
    # Make Sector + exchange categorical
    dataset.onehot_encode(["Sector", "Exchange"])

    dataset.rolling_apply(lambda x: x.std(), window=7 * 20, on_cols=["Open"], new_names=["Volatility"], drop_col = False)
    dataset.rolling_apply(lambda x: x.mean(), window=7 * 20, on_cols=["Volume"], new_names=["Rolling Volume"], drop_col = False)
    dataset.save_data('Data/tables/snp500.csv')
    
    features = ["Volume", "Volatility", "^VIX", "Rolling Volume"] # ADD ONEHOT ENCODINGS
    features += [col for col in dataset.data.columns if col.startswith("Sector_") or col.startswith("Exchange_")]
    daily_features = ["Volatility", "^VIX"]
    daily_features += [col for col in dataset.data.columns if col.startswith("Sector_") or col.startswith("Exchange_")]

    dataset.prepare_features(features=features, target="Volume", shift_features=True, daily_features=daily_features)


    print(data)
    exit(1)