import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 0. Loading and Normalizing the Dataset =====
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data.csv').as_posix()
data = pd.read_csv(file_path)

# Plotting the dataset and financial indicators
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['MA_50'], label='MA 50', color='orange')
plt.plot(data['MA_200'], label='MA 200', color='purple')
plt.plot(data['EMA_26'], label='EMA 50', color='green')
plt.plot(data['EMA_200'], label='EMA 200', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Close Price and Financial Indicators Over Time')
plt.legend()
plt.grid(True)
plt.show()