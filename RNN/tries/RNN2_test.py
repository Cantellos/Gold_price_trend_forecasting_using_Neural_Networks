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
seq_length = 30  # number of time steps in each input sequence
pred_length = 7  # number of future steps to predict

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

# Print x and y shapes
#print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")

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
hidden_size = 128     # Hidden layer size
num_layers = 1       # Number of RNN layers
output_size = pred_length      # Output is the next time step for the last column (target)
lr = 0.002           # Learning rate
num_epochs = 200     # Number of epochs
patience = 20        # Early stopping patience
model = RNNModel(input_size, hidden_size, num_layers, output_size)

# ---- Step 7: Loss Function and Optimizer ----
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- Step 8: Training Loop ----
def train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs, patience):

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train)
        #if epoch == 0: print(f"outputs shape {outputs.shape}, y_train shape {y_train.shape}")
        # Compute the loss
        loss = criterion(outputs, y_train)  # Squeeze to remove the extra dimension in output
        train_losses.append(loss.item())
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Early stopping condition
        if patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                break

    return train_losses, val_losses

train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs, patience)

# TODO: add testing e graphics

# ---- Step 9: Plot the Training and Validation Losses ----
plt.figure(figsize=(8, 5))
plt.plot(range(3, len(train_losses) + 1), train_losses[2:], label='Train Loss', marker='o')
plt.plot(range(3, len(val_losses) + 1), val_losses[2:], label='Validation Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_curve_path = (Path(__file__).resolve().parent.parent / 'images' / 'RNN_test_loss_curve.png').as_posix()
Path(loss_curve_path).parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
plt.savefig(loss_curve_path)
plt.show()

# ---- Step 10: Testing the Model ----
model.eval()
with torch.no_grad():
    # Make predictions on the test set
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)  # Squeeze to match the dimensions
    print(f"MSE Test Loss: {test_loss.item():.4f}")

# ---- Step 11: Inverse Normalization for Prediction Visualization ----
# Rescale predictions back to original scale
predictions_rescaled = target_scaler.inverse_transform(predictions.numpy())
y_test_rescaled = target_scaler.inverse_transform(y_test.numpy())

print(f"Predictions shape: {predictions_rescaled.shape}, y_test shape: {y_test_rescaled.shape}")

# ---- Step 12: Calculate Accuracy Loss Based on a Threshold ----
threshold = 5  # % of tolerance
corrects = 0

# Calculate the number of correct predictions within the threshold
for length in range(len(predictions)):
    for i in (0, pred_length-1):
        if abs(predictions[length][i] - y_test[length][i]) <= threshold/100*y_test[length][i]:
            corrects += 1

# Calculate the loss as the ratio of incorrect predictions
accuracy = corrects / (len(predictions) * pred_length)
print(f'\nAccuracy - Test set (MLP): {accuracy*100:.4f}% of correct predictions within {threshold}%\n')

# TODO: fix Graphic Visualization for a 5-step prediction

#"""
# ---- Step 13: Visualize Predictions vs. Ground Truth ----
# Plot Actual vs Predicted Prices

# Supponiamo che tu abbia già ottenuto:
# - y_test: ground truth [N, 7]
# - test_predictions: predizioni [N, 7]
# - target_scaler: scaler già fit-ato, utile per riportare i dati ai valori originali

# Inizializza array per visualizzare le previsioni continue
full_len = len(predictions_rescaled) + 6  # perché ogni predizione è lunga 7
pred_line = np.zeros(full_len)
real_line = np.zeros(full_len)
counts = np.zeros(full_len)  # per la media in caso di sovrapposizione

# Inserisci ogni previsione nel punto corretto
for i in range(len(predictions_rescaled)):
    for j in range(7):  # 7 = pred_length
        pred_line[i + j] += predictions_rescaled[i][j]
        real_line[i + j] += y_test_rescaled[i][j]
        counts[i + j] += 1

# Media dei valori sovrapposti
pred_line /= counts
real_line /= counts

# Plot
plt.figure(figsize=(14, 6))
plt.plot(real_line, label='Valori reali', color='red')
plt.plot(pred_line, label='Valori predetti', color='blue')
plt.legend()
plt.title("Predizione continua su 7 giorni per sequenze multiple")
plt.xlabel("Tempo (giorni)")
plt.ylabel("Prezzo")
plt.grid(True)
plt.show()
#"""

# Save the model in the "models" folder
model_path = (Path(__file__).resolve().parent.parent / 'models' / 'RNN2_test_model.pth').as_posix()
Path(model_path).parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
torch.save(model.state_dict(), model_path)
