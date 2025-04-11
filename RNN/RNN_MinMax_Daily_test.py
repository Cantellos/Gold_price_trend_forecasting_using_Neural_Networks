import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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

# Split dataset (70% train, 15% val, 15% test)
train_size = int(len(data) * 0.7)   
val_size = int(len(data) * 0.15)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Normalize features feature by feature
scaler = MinMaxScaler()

# Fit only on the training set
scaler.fit(train_data[features])

# Transform training, validation, and test using the same scaler
train_data[features] = scaler.transform(train_data[features])
val_data[features] = scaler.transform(val_data[features])
test_data[features] = scaler.transform(test_data[features])


# ===== 2. Definition of the MLP (Fully Connected Layer) =====
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1])  # Adjust indexing to match the tensor's dimensions
        return out


# Set hyperparameters and instantiate the model
input_size = len(features)
hidden_size = 64
num_layers = 1
output_size = 1
lr = 0.001

# Instantiate the model, define loss function and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr)


# ===== 3. Training Function =====
def train_model(model, train_X, train_y, val_X, val_y, criterion, optimizer, num_epochs, batch_size):
    train_losses = []
    val_losses = []
    patience = 5  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i in range(0, len(train_X), batch_size):
            x_train = torch.tensor(train_data[features].values, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
            y_train = torch.tensor(train_data[target].values, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_X) // batch_size)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_X), batch_size):
                x_val = torch.tensor(val_data[features].values, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
                y_val = torch.tensor(val_data[target].values, dtype=torch.float32).unsqueeze(1)

                output = model(x_val)
                loss = criterion(output, y_val)
                val_loss += loss.item()

            val_losses.append(val_loss / len(val_X) // batch_size)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')

        # Early stopping condition
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    return train_losses, val_losses


# ===== 4. Training the Model =====
num_epochs = 150
batch_size=32
train_losses, val_losses = train_model(model, train_data[features], train_data[target], val_data[features], val_data[target], criterion, optimizer, num_epochs, batch_size)


# ===== 5. Plotting the Losses =====
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (excluding first 1 for graphic reasons)')
plt.show()


# ===== 6. Testing the Model =====
model.eval()
test_loss = 0.0
predictions = []
actuals = []
# MSE Loss
with torch.no_grad():
    for i in range(len(test_data)):
        x_test = torch.tensor(test_data[features].values, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
        y_test = torch.tensor(test_data[target].values, dtype=torch.float32).unsqueeze(1)

        output = model(x_test)
        loss = criterion(output, y_test)
        test_loss += loss.item()

        predictions.append(output.item())
        actuals.append(y_test.item())

final_test_loss = test_loss / len(test_data)
print(f'\nMSE Loss - Test set (MLP): {final_test_loss:.6f}')

# Accuracy Loss
def accuracy_based_loss(predictions, targets, threshold):
    accuracy = 0
    corrects = 0
    # Calculate the number of correct predictions within the threshold
    for length in range(len(predictions)):
        if abs(predictions[length] - targets[length]) <= threshold*targets[length]:
            corrects += 1
    # Calculate the loss as the ratio of incorrect predictions
    accuracy = corrects / len(predictions)
    return accuracy

loss = accuracy_based_loss(predictions, actuals, threshold=0.02)  # 2% tolerance
print(f'\nAccuracy - Test set (MLP): {loss*100:.4f}% of correct predictions within 2%\n')

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