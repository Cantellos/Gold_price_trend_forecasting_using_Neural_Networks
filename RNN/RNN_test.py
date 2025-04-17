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

training= data[:train_size]
validation = data[train_size:train_size + val_size]
testing = data[train_size + val_size:]

# Normalize features using Min-Max scaling
scaler = MinMaxScaler()

# Fit only on the training set, but transform all of them using the same scaler
scaler.fit(training[features])

train_data = scaler.transform(training[features])
val_data = scaler.transform(validation[features])
test_data = scaler.transform(testing[features])

# Normalize target variable (future_close) separately
scaler.fit(training[[target]])

train_target = scaler.transform(training[[target]])
val_target = scaler.transform(validation[[target]])
test_target = scaler.transform(testing[[target]])

# Convert data to PyTorch tensors
def create_tensor_dataset(data, target):
    # Add dimension to ensure the correct shape for RNN input
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
    y = torch.tensor(target, dtype=torch.float32)
    return x, y

train_x, train_y = create_tensor_dataset(train_data, train_target)
val_x, val_y = create_tensor_dataset(val_data, val_target)
test_x, test_y = create_tensor_dataset(test_data, test_target)


# ===== 2. Definition of the MLP (Fully Connected Layer) =====
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Set hyperparameters and instantiate the model
input_size = len(features)
hidden_size = 128
num_layers = 1
output_size = 1
lr = 0.001

# Instantiate the model, define loss function and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size)
model = model.to(device)

criterion = nn.MSELoss()        # Mean Squared Error: sensibile agli outliers, per non sbagliare mai troppo
#criterion = nn.SmoothL1Loss() # Huber Loss: robusto agli outliers, ma meno sensibile ai picchi rispetto all'MSE

optimizer = optim.RMSprop(model.parameters(), lr)
# optimizer = optim.Adam(model.parameters(), lr)


# ===== 3. Training Function =====
def train_model(model, train_x, train_y, val_x, val_y, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    patience = 20 # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

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
train_losses, val_losses = train_model(model, train_x, train_y, val_x, val_y, criterion, optimizer, num_epochs)


# ===== 5. Plotting the Losses =====
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


# ===== 6. Testing the Model =====
model.eval()
test_loss = 0.0
predictions = []
actuals = []
# MSE Loss
with torch.no_grad():
    for i in range(len(test_data)):

        output = model(test_x[i].unsqueeze(0))  # Add batch dimension
        loss = criterion(output, test_y[i].unsqueeze(0))  # Add batch dimension
        test_loss += loss.item()

        predictions.extend(output)
        actuals.extend(test_y[i])

final_test_loss = test_loss / len(test_data)
print(f'\nMSE Loss - Test set (MLP): {final_test_loss:.6f}')

# Accuracy Loss
def accuracy_based_loss(predictions, actuals, threshold):
    accuracy = 0
    corrects = 0
    # Calculate the number of correct predictions within the threshold
    for length in range(len(predictions)):
        if abs(predictions[length] - actuals[length]) <= threshold*actuals[length]:
            corrects += 1
    # Calculate the loss as the ratio of incorrect predictions
    accuracy = corrects / len(predictions)
    return accuracy

loss = accuracy_based_loss(predictions, actuals, threshold=0.02)  # 2% tolerance
print(f'\nAccuracy - Test set (MLP): {loss*100:.4f}% of correct predictions within 2%\n')

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actual Price", color='blue')
plt.plot(predictions, label="Predicted Price", color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted Price (Test Set)")
plt.legend()
plt.grid(True)
plt.show()

# Save the model in the "models" folder
model_path = (Path(__file__).resolve().parent.parent / 'models' / 'RNN_test_model.pth').as_posix()
Path(model_path).parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
torch.save(model.state_dict(), model_path)