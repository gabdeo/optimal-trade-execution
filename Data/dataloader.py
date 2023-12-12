import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

class Dataset():

    def __init__(self, tickers: list[str], start_date: str, end_date: str, interval="1h", date_fromat='%Y-%m-%d'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.date_format = date_fromat

    def load_data(self, local_path = None):

        if local_path is not None:
            data = pd.read_csv(local_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'], format=self.date_format)
            data = data.set_index('Datetime')
            
        else:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date, interval=self.interval)
        
        self.data = data
        return data
    
    def save_data(self, path):
        self.data.to_csv(path)


if __name__ == '__main__':
    
    with open('Data/tickers/snp500.txt', 'r') as f:
        tickers = f.read().splitlines()
    start_date = "2022-01-01"
    end_date = "2023-12-10"

    dataset = Dataset(tickers, start_date, end_date)
    data = dataset.load_data()

    print(data)
    exit(1)