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

    def load_data(self, local_path = None, metrics = ['Open', 'Volume', 'Dividends', 'Stock Splits'], preprocess = True):

        if local_path is not None:
            data = pd.read_csv(local_path, index_col=0, parse_dates=True, date_format=self.date_format, header=[0,1])

            if not all([m in data.columns for m in metrics]):
                raise ValueError("Metrics are not in the loaded data")
            data = data[metrics]
            
            
        else:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date, interval=self.interval, actions=True)
            data = data.stack(level=1) # Columns: features, Rows: date, ticker

            if preprocess:
                self.data = data
                data = self.preprocess_data()

            data = data[metrics]
            data = data.swaplevel().sort_index() # Sort by ticker then by date
        
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
            raise warn("NaN values remain in the dataset.")

        return self.data
    
    
    


if __name__ == '__main__':
    
    with open('Data/tickers/snp500.txt', 'r') as f:
        tickers = f.read().splitlines()
    start_date = "2022-01-01"
    end_date = "2023-12-10"

    dataset = DataLoader(tickers, start_date, end_date)
    data = dataset.load_data()
    
    alt_data = pd.read_excel('Data/alt_features/industries_exchanges.xlsx', index_col=1)
    data = pd.merge(data.reset_index(), alt_data.reset_index(), how="left", left_on="level_0", right_on='Symbol')
    data = data.set_index(['level_0', 'level_1'])
    data.index.names = ['Ticker', 'Date']

    
    dataset.save_data('Data/tables/snp500.csv')

    print(data)
    exit(1)