import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from Data import DataLoader
import tqdm
from matplotlib import pyplot as plt

class SimpleFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(SimpleFNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class SimpleFNNModel:
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        self.model = SimpleFNN(input_size, hidden_size, output_size, n_layers)
        self.xscaler, self.yscaler = StandardScaler(), StandardScaler()
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, df_train, target_columns, batch_size=32, epochs=100):
        data_loader = self._prepare_data(df_train, target_columns, batch_size, train=True)

        for epoch in range(epochs):
            self.model.train()
            for X_batch, y_batch in tqdm.tqdm(data_loader):
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f'Train Loss: {loss.item():.4f}')


    def evaluate(self, df_test, target_columns):
        X_test, y_test = self._prepare_data(df_test, target_columns, train=False)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test)
            test_loss = self.criterion(y_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
        return y_pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def _prepare_data(self, df, target_columns, batch_size, train=True):
        X = df[df.columns.difference(target_columns)]
        y = df[target_columns]

        if train:
            X_scaled = self.xscaler.fit_transform(X)
            # y_scaled = self.yscaler.fit_transform(y)
        else:
            X_scaled = self.xscaler.transform(X)
            # y_scaled = self.yscaler.transform(y)

        data = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32))
        data_loader = TorchDataLoader(data, batch_size=batch_size, shuffle=train)
        return data_loader

if __name__ == "__main__":
    
    epochs = 128
    n_layers = 3
    hidden_size = 32
    batch_size = 128
    train_test_split = 0.8
    load = True

    if not load:
        with open('Data/tickers/snp500.txt', 'r') as f:
            tickers = f.read().splitlines()
        tickers = tickers[:5]
        start_date = "2022-01-01"
        end_date = "2023-12-10"

        dataset = DataLoader(tickers, start_date, end_date)
        dataset.load_data('Data/tables/snp500.csv')

        features = ["Volume", "Volatility", "^VIX", "Rolling Volume"] # ADD ONEHOT ENCODINGS
        features += [col for col in dataset.data.columns if col.startswith("Sector_") or col.startswith("Exchange_")]
        daily_features = ["Volatility", "^VIX", "Rolling Volume"]
        daily_features += [col for col in dataset.data.columns if col.startswith("Sector_") or col.startswith("Exchange_")]

        dataset.prepare_features(features=features, target="Volume", shift_features=True, daily_features=daily_features)

        data = dataset.data
        data = data.swaplevel().sort_index().swaplevel()
        train, test = data[:int(len(data) * train_test_split)], data[int(len(data) * train_test_split):]
        train.to_csv('Data/tables/snp500_train.csv')
        test.to_csv('Data/tables/snp500_test.csv')
    
    else:
        train = pd.read_csv('Data/tables/snp500_train.csv', index_col=[0, 1])
        test = pd.read_csv('Data/tables/snp500_test.csv', index_col=[0, 1])

    features_columns = [c for c in train.columns if not c.startswith("Target")]
    target_columns = [c for c in train.columns if c.startswith("Target")]
    
    log_cols = [c for c in train.columns if "Volume" in c or c.startswith("Target")]
    train[log_cols] = train[log_cols].where(train[log_cols] > 0, 1)
    test[log_cols] = test[log_cols].where(test[log_cols] > 0, 1)
    train[log_cols] = train[log_cols].apply(lambda x: np.log(x))
    test[log_cols] = test[log_cols].apply(lambda x: np.log(x))

    model = SimpleFNNModel(len(features_columns), hidden_size, len(target_columns), n_layers)
    model.train(train, target_columns=target_columns, batch_size = batch_size, epochs=epochs)
    model.evaluate(test, target_columns=target_columns)
    model.save_model('Models/simple_fnn.pt')

    exit(1)