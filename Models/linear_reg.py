import pandas as pd
import numpy as np
from Data import DataLoader
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np


class LassoModel:
    def __init__(self, alpha=1):
        self.model = Lasso(alpha=alpha) 

    def train(self, df_train, target_columns):
        features = df_train.columns.difference(target_columns)
        target = df_train[target_columns]

        self.model.fit(df_train[features], target)

    def evaluate(self, df_test, target_columns):
        features = df_test.columns.difference(target_columns)
        target = df_test[target_columns]
        
        results = {t:[] for t in target_columns}

        y_pred = self.model.predict(df_test[features])
        test_loss = mean_squared_error(target, y_pred)

        print(f'Test Loss: {test_loss:.4f}')
        return y_pred

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


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

log_cols = [c for c in train.columns if "Volume" in c or c.startswith("Target")]
train[log_cols] = train[log_cols].where(train[log_cols] > 0, 1)
test[log_cols] = test[log_cols].where(test[log_cols] > 0, 1)
train[log_cols] = train[log_cols].apply(lambda x: np.log(x))
test[log_cols] = test[log_cols].apply(lambda x: np.log(x))

target_columns = [c for c in train.columns if c.startswith("Target")]

model = LassoModel(alpha=0.5)
model.train(train, target_columns=target_columns)
preds = model.evaluate(test, target_columns=target_columns)
preds = pd.DataFrame(preds, index=test.index, columns = target_columns)
preds.to_csv('Data/predictions/preds.csv')

exit(1)
