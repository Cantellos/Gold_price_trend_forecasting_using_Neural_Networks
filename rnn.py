import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Load and preprocess the dataset
file_path = (Path(__file__).resolve().parent / '.data' / 'dataset' / 'XAU_1h_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Normalize relevant columns
scaler = MinMaxScaler()
columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', 'RSI']
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Add target variable and drop NaN
data['future_close'] = data['Close'].shift(-1)
data = data.dropna()

# Split data
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Prepare tensors
train_features = torch.tensor(train_data[columns_to_normalize].values, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_data['future_close'].values, dtype=torch.float32).unsqueeze(1)

val_features = torch.tensor(val_data[columns_to_normalize].values, dtype=torch.float32).unsqueeze(1)
val_labels = torch.tensor(val_data['future_close'].values, dtype=torch.float32).unsqueeze(1)

test_features = torch.tensor(test_data[columns_to_normalize].values, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_data['future_close'].values, dtype=torch.float32).unsqueeze(1)

# Define the GRU model
class GRUPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUPricePredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_size = len(columns_to_normalize)
hidden_size = 64
num_layers = 2
output_size = 1

model = GRUPricePredictor(input_size, hidden_size, num_layers, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_loss = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Validation and test evaluation
model.eval()
with torch.no_grad():
    val_outputs = model(val_features)
    test_outputs = model(test_features)
    val_loss = criterion(val_outputs, val_labels).item()
    test_loss = criterion(test_outputs, test_labels).item()

print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")