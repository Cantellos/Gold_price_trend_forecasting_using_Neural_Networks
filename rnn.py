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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Convert outputs to numpy for metrics calculations
val_outputs = model(val_features.unsqueeze(1)).detach().numpy()
test_outputs = model(test_features.unsqueeze(1)).detach().numpy()
val_labels_np = val_labels.numpy()
test_labels_np = test_labels.numpy()

# --- 1. Mean Absolute Error (MAE) ---
val_mae = mean_absolute_error(val_labels_np, val_outputs)
test_mae = mean_absolute_error(test_labels_np, test_outputs)
print(f'Validation MAE: {val_mae:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# --- 2. R-squared (R²) ---
val_r2 = r2_score(val_labels_np, val_outputs)
test_r2 = r2_score(test_labels_np, test_outputs)
print(f'Validation R²: {val_r2:.4f}')
print(f'Test R²: {test_r2:.4f}')

# --- 3. Root Mean Squared Error (RMSE) ---
val_rmse = np.sqrt(mean_squared_error(val_labels_np, val_outputs))
test_rmse = np.sqrt(mean_squared_error(test_labels_np, test_outputs))
print(f'Validation RMSE: {val_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

# --- 4. Prediction Plot (True vs Predicted) ---
plt.figure(figsize=(12, 6))
plt.plot(test_labels_np, label='True Values', color='blue')
plt.plot(test_outputs, label='Predicted Values', color='orange', alpha=0.7)
plt.title('True vs Predicted Prices (Test Set)')
plt.legend()
plt.show()

# --- 5. Prediction Errors (Residuals) ---
errors = test_labels_np - test_outputs
plt.figure(figsize=(12, 6))
plt.plot(errors, color='red', alpha=0.6)
plt.title('Prediction Errors (Residuals)')
plt.xlabel('Sample Index')
plt.ylabel('Error')
plt.show()

# --- 6. Correlation Between Predicted and True Values ---
correlation = np.corrcoef(test_labels_np.flatten(), test_outputs.flatten())[0, 1]
print(f'Correlation between true and predicted values (Test Set): {correlation:.4f}')

# --- 7. Learning Curves ---
# Assuming you have train_loss and validation_loss tracked during training
if 'train_loss' in locals() and 'validation_loss' in locals():
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss', color='green')
    plt.plot(validation_loss, label='Validation Loss', color='purple')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    print("Learning curves are unavailable because 'train_loss' and/or 'validation_loss' were not tracked.")

# --- Summary ---
print("\nSummary of Metrics:")
print(f"Validation: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R²={val_r2:.4f}")
print(f"Test: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
print(f"Correlation (Test Set): {correlation:.4f}")

