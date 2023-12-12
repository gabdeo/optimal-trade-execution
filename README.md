# trade-prescription
Prediction/prescription modelling for large trade order execution


- Discussion: Some data is missing (68 tickers couldn't be downloaded)- MAR or not? discuss other biases in data
- Filling nan values: don't use fancy methods (like knn) bc data is probably very noisy and this could lead to lookahead bias

Prediction
- features: recent vol (price squared), industry, exchange, previous volume (rolling?), vix