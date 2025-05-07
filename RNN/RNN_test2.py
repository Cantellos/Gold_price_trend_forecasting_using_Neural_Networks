import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Imposta il dispositivo per l'esecuzione su GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pd.options.mode.copy_on_write = True


# ===== 1. Caricamento e Normalizzazione del Dataset =====
# Load the dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Separate features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# ---- Step 2: Normalize Features and Target Separately ----
# Separate the features (first 11 columns) from the target (12th column)
input_data = data[features]  # shape: [3000, 11]
target_data = data[target].values.reshape(-1, 1)  # shape: [3000, 1]

# Normalize the features (first 11 columns) between 0 and 1
feature_scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = feature_scaler.fit_transform(input_data)

# Normalize the target (last column) between 0 and 1
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = target_scaler.fit_transform(target_data)

# Combine the scaled features and target back together
data_scaled = np.hstack((features_scaled, target_scaled))

# ---- Step 3: Prepare Data Sequences ----
seq_length = 50  # number of time steps in each input sequence
pred_length = 5  # number of future steps to predict

X = []
y = []

max_start = len(data_scaled) - seq_length - pred_length + 1
for i in range(max_start):
    input_seq = data_scaled[i : i + seq_length, :-1]  # First 11 columns (input features)
    target_seq = data_scaled[i + seq_length : i + seq_length + pred_length, -1]  # Last column (target to predict)
    
    X.append(input_seq)
    y.append(target_seq)

# Convert lists to numpy arrays
X = np.array(X)  # shape: [N, 50, 11]
y = np.array(y)  # shape: [N, 5]

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# ---- Step 4: Train-Validation-Test Split ----
# (70% train, 15% val, 15% test)
train_size = int(len(data) * 0.7)   
val_size = int(len(data) * 0.15)

X_train = X[:train_size]
X_val = X[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_train = y[:train_size]
y_val = y[train_size:train_size + val_size]
y_test = y[train_size + val_size:]

# TODO: continua da qua !!!

# ---- Step 5: Define the RNN Model ----
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, _ = self.rnn(x)  # out: [batch_size, seq_len, hidden_size]
        last_output = out[:, -1, :]  # take the last time step
        return self.fc(last_output)  # output: [batch_size, output_size]

# ---- Step 6: Model Initialization and Hyperparameters ----
input_size = len(features)     # First 11 columns (input features)
hidden_size = 64     # Hidden layer size
num_layers = 2       # Number of RNN layers
output_size = 5      # Output is the next time step for the last column (target)

model = RNNModel(input_size, hidden_size, num_layers, output_size)

# ---- Step 7: Loss Function and Optimizer ----
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- Step 8: Training Loop ----
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    if epoch == 0: print(f"outputs shape {outputs.shape}, y_train shape {y_train.shape}")
    # Compute the loss
    loss = criterion(outputs, y_train)  # Squeeze to remove the extra dimension in output
    train_losses.append(loss.item())

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ---- Step 9: Evaluate the Model ----
model.eval()
with torch.no_grad():
    # Make predictions
    train_predictions = model(X_train)
    test_predictions = model(X_test)

    # Compute test loss
    test_loss = criterion(test_predictions, y_test)  # Squeeze to match the dimensions
    print(f"Test Loss: {test_loss.item():.4f}")

# ---- Step 10: Plot the Training Loss ----
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# ---- Step 11: Inverse Normalization for Prediction Visualization ----
# Rescale predictions back to original scale
train_predictions_rescaled = target_scaler.inverse_transform(train_predictions.numpy())
y_train_rescaled = target_scaler.inverse_transform(y_train.numpy())

test_predictions_rescaled = target_scaler.inverse_transform(test_predictions.numpy())
y_test_rescaled = target_scaler.inverse_transform(y_test.numpy())

# ---- Step 12: Visualize Predictions vs. Ground Truth ----
# Plot a few samples from the training set
plt.figure(figsize=(12, 6))
plt.plot(train_predictions_rescaled[0], label="Predicted", color='blue')
plt.plot(y_train_rescaled[0], label="Actual", color='red')
plt.title("Predicted vs Actual (First Sample from Training)")
plt.legend()
plt.show()

# Plot a few samples from the test set
plt.figure(figsize=(12, 6))
plt.plot(test_predictions_rescaled[0], label="Predicted", color='blue')
plt.plot(y_test_rescaled[0], label="Actual", color='red')
plt.title("Predicted vs Actual (First Sample from Test)")
plt.legend()
plt.show()
