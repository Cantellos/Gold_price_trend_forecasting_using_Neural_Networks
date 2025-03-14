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

# Split data in training, validation and test sets
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Define the features to be standardized
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_close']

print('Train data shape:', val_data.shape)
print('Train data examples:')
print(val_data.head())

# Normalization of data 
scaler = MinMaxScaler()
train_data[features] = scaler.fit_transform(train_data[features])
val_data[features] = scaler.transform(val_data[features])
test_data[features] = scaler.transform(test_data[features])

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
        out = self.fc(out[:, -1, :])
        return out

# Define the model, loss function and optimizer
input_size = len(features) - 1  # Number of features
hidden_size = 50  # Number of features in hidden state
num_layers = 2  # Number of stacked RNN layers
output_size = 1  # Output size

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the training function
def train_model(model, train_data, val_data, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i in range(len(train_data) - 1):
            x_train = torch.tensor(train_data[features].iloc[i].values[:-1], dtype=torch.float32).unsqueeze(0)
            y_train = torch.tensor(train_data['future_close'].iloc[i], dtype=torch.float32).unsqueeze(0)
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_data))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_data) - 1):
                x_val = torch.tensor(val_data[features].iloc[i].values[:-1], dtype=torch.float32).unsqueeze(0)
                y_val = torch.tensor(val_data['future_close'].iloc[i], dtype=torch.float32).unsqueeze(0)
                output = model(x_val)
                loss = criterion(output, y_val)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_data))

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


