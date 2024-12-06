import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt

# Load and preprocess the dataset
file_path = (Path(__file__).resolve().parent / '.data' / 'dataset' / 'XAU_1w_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Drop useless coloumns (Date and Time), add target variable and drop NaN
data['future_close'] = data['Close'].shift(-1)
data = data.dropna()
data = data.drop(['Date', 'Time'], axis=1)

# Split data in training, validation and test sets
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply StandardScaler to each feature separately
train_data[features] = scaler.fit_transform(train_data[features])
val_data[features] = scaler.transform(val_data[features])  # Use the same scaler to transform val and test
test_data[features] = scaler.transform(test_data[features])

# If you want to check the standardized data
print("Standardized Train Data:")
print(train_data.head())

print("\nStandardized Validation Data:")
print(val_data.head())

print("\nStandardized Test Data:")
print(test_data.head())


