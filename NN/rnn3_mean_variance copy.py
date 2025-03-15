import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# Load and preprocess the dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_4h_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Separate features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Split the dataset using the expanding window method
initial_train_size = int(0.3 * len(data))
train_data = data.iloc[:initial_train_size]
val_data = data.iloc[initial_train_size:int(0.7 * len(data))]
test_data = data.iloc[int(0.7 * len(data)):]

# Dizionari per salvare i valori di media e varianza di ogni feature
train_mean = {}
train_variance = {}
train_std = {}

for feature in features:
    # Calcola media e varianza per ogni singola feature
    train_mean[feature] = train_data[feature].sum() / len(train_data)  # Media
    train_variance[feature] = ((train_data[feature] - train_mean[feature]) ** 2).sum() / len(train_data)  # Varianza
    train_std[feature] = np.sqrt(train_variance[feature])  # Deviazione standard

# Normalizzazione di ogni feature separatamente
for feature in features:
    train_data[feature] = (train_data[feature] - train_mean[feature]) / train_std[feature]
    val_data[feature] = (val_data[feature] - train_mean[feature]) / train_std[feature]
    test_data[feature] = (test_data[feature] - train_mean[feature]) / train_std[feature]

# Normalizzazione del target separatamente
train_target_mean = train_data[target].sum() / len(train_data)
train_target_variance = ((train_data[target] - train_target_mean) ** 2).sum() / len(train_data)
train_target_std = np.sqrt(train_target_variance)

train_data[target] = (train_data[target] - train_target_mean) / train_target_std
val_data[target] = (val_data[target] - train_target_mean) / train_target_std
test_data[target] = (test_data[target] - train_target_mean) / train_target_std

print(train_data.head())
print(train_data[features].min())
print(train_data[features].max())

# Convert data to PyTorch tensors
def create_tensor_dataset(df, features, target):
    X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
    y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)  # Make sure y has correct shape
    return X, y

train_X, train_y = create_tensor_dataset(train_data, features, target)
val_X, val_y = create_tensor_dataset(val_data, features, target)
test_X, test_y = create_tensor_dataset(test_data, features, target)

print(train_X.shape, train_y.shape)
print("OK")

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

# Define the model, loss function, and optimizer
input_size = len(features)
hidden_size = 64
num_layers = 1
output_size = 1
lr=0.1
num_epochs=100

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

# Define the training function
def train_model(model, train_X, train_y, val_X, val_y, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(val_X)
            val_loss = criterion(val_output, val_y)
            val_losses.append(val_loss.item())

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_X, train_y, val_X, val_y, criterion, optimizer, num_epochs)

# Plot the training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the test set
def evaluate_model(model, test_X, test_y, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i in range(len(test_X)):
            x_test = test_X[i].unsqueeze(0)  # Add batch dimension
            y_test = test_y[i].unsqueeze(0)  # Add batch dimension
            output = model(x_test)
            loss = criterion(output, y_test)
            test_loss += loss.item()
    
    test_loss /= len(test_X)  # Compute average loss
    return test_loss

# Compute test loss after training
test_loss = evaluate_model(model, test_X, test_y, criterion)
print(f"Final Test Loss: {test_loss:.4f}")