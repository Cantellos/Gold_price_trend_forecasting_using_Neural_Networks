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

# Drop useless coloumns (Date and Time), add Target variable (shifted because it has to be the future price, not current) and drop NaN values
data = data.drop(['Date', 'Time'], axis=1)
data['future_close'] = data['Close'].shift(-1)
data = data.dropna()

# Define the initial training size
initial_train_size = int(0.3 * len(data))  # Start with 30% of data for training

# Expanding Window Cross-Validation
window_splits = []
scalers = {}

for i in range(initial_train_size, len(data) - 1, int(0.05 * len(data))):  # Expand by 5% each step
    train_set = data.iloc[:i]  # Train on the expanding window
    test_set = data.iloc[i:i + int(0.05 * len(data))]  # Test on the next chunk

    # Apply MinMaxScaler only on training data to avoid data leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_set)
    test_scaled = scaler.transform(test_set)

    # Store results
    window_splits.append((train_scaled, test_scaled))
    scalers[i] = scaler  # Save the scaler for inverse transformation

# Print the first three expanding windows (for verification)
for i, (train, test) in enumerate(window_splits[:3]):
    print(f"Window {i+1}: Train Size: {len(train)}, Test Size: {len(test)}")