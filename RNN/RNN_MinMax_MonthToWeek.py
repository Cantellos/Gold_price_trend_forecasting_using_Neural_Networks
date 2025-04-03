import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Define the dataset processing function
def create_sequences(data, seq_length=30, pred_length=7):
    sequences, targets = [], []
    for i in range(len(data) - seq_length - pred_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length:i + seq_length + pred_length, -1])
    return np.array(sequences), np.array(targets)

# Load and preprocess the dataset --------------------------------------------
# Load the dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Separate features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Normalize dataset
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
seq_length = 30  # One month of data
pred_length = 7  # Predict one week
sequences, targets = create_sequences(data_scaled, seq_length, pred_length)

# Split dataset
train_size = int(0.7 * len(sequences))
val_size = int(0.15 * len(sequences))

train_X, train_Y = sequences[:train_size], targets[:train_size]
val_X, val_Y = sequences[train_size:train_size + val_size], targets[train_size:train_size + val_size]
test_X, test_Y = sequences[train_size + val_size:], targets[train_size + val_size:]

# Convert to tensors
train_X, train_Y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_Y, dtype=torch.float32)
val_X, val_Y = torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_Y, dtype=torch.float32)
test_X, test_Y = torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_Y, dtype=torch.float32)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use the last time step output
        return out

# Model parameters
input_size = data.shape[1]  # Number of features
hidden_size = 64
num_layers = 2
output_size = pred_length  # Predict a sequence of 7 closing prices

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_X)
    loss = criterion(output, train_Y)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_output = model(val_X)
        val_loss = criterion(val_output, val_Y)
        val_losses.append(val_loss.item())
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Inverse transform the normalized values back to original price scale
def inverse_transform(preds, min_val, max_val):
    return (preds + 1) * (max_val - min_val) / 2 + min_val

# Inverse transform the predictions for graphical evaluation ------------------------------------------
# Get predictions on test set
model.eval()
predictions = []
actual_values = []

with torch.no_grad():
    for i in range(len(test_X)):
        x_test = test_X[i].unsqueeze(0)
        y_test = test_Y[i].unsqueeze(0)
        pred = model(x_test)
        predictions.append(pred.item())
        actual_values.append(y_test.item())

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label="Actual Price", color='blue')
plt.plot(predictions, label="Predicted Price", color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted Price (Test Set)")
plt.legend()
plt.grid(True)
plt.show()