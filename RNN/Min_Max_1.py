import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# Load and preprocess the dataset --------------------------------------------
# Load the dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Separate features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Split the dataset using the expanding window method
initial_train_size = int(0.3 * len(data))
train_data = data.iloc[:initial_train_size]
val_data = data.iloc[initial_train_size:int(0.7 * len(data))]
test_data = data.iloc[int(0.7 * len(data)):]



# Normalize the data using Min-Max scaling ------------------------------------
# Dictionaries to store min and max values for each feature
train_min = {}
train_max = {}

# Compute min and max for each feature based only on training data
for feature in features:
    train_min[feature] = train_data[feature].min()
    train_max[feature] = train_data[feature].max()

# Apply Min-Max scaling to each feature (scales data to range [-1, 1])
for feature in features:
    train_data[feature] = 2 * (train_data[feature] - train_min[feature]) / (train_max[feature] - train_min[feature]) - 1
    val_data[feature] = 2 * (val_data[feature] - train_min[feature]) / (train_max[feature] - train_min[feature]) - 1
    test_data[feature] = 2 * (test_data[feature] - train_min[feature]) / (train_max[feature] - train_min[feature]) - 1

# Normalize target variable separately using Min-Max Scaling
target_min = train_data[target].min()
target_max = train_data[target].max()

train_data[target] = 2 * (train_data[target] - target_min) / (target_max - target_min) - 1
val_data[target] = 2 * (val_data[target] - target_min) / (target_max - target_min) - 1
test_data[target] = 2 * (test_data[target] - target_min) / (target_max - target_min) - 1



# Implement Sliding Window Input ---------------------------------------------
# Function to create sliding window sequences
def create_sequences(df, features, target, seq_length):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df[features].iloc[i:i+seq_length].values)  # Past `seq_length` days
        y.append(df[target].iloc[i+seq_length])  # Target value for the next day
    return np.array(X), np.array(y)

sequence_length = 10  # Number of past days used to predict the next day

# Create tensor datasets with sliding window input
train_X, train_y = create_sequences(train_data, features, target, sequence_length)
val_X, val_y = create_sequences(val_data, features, target, sequence_length)
test_X, test_y = create_sequences(test_data, features, target, sequence_length)



# Convert data to PyTorch tensors --------------------------------------------
train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1) 

val_X = torch.tensor(val_X, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1)

test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1)



# Define the RNN model -------------------------------------------------------
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

class RNN_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(RNN_dropout, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        # Define the RNN layer with dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout_prob)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)
        
        # Dropout layer (applied to the output of the RNN layer)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out


# Set hyperparameters and instantiate the model ------------------------------
input_size = train_X.shape[2] # Exclude the target variable
hidden_size = 128
num_layers = 5
num_epochs=50
output_size = 1
lr=0.001
#batch_size=64

# Instantiate the model, define loss function and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)



# Define the training function with early stopping ----------------------------
def train_model(model, train_X, train_y, val_X, val_y, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    
    early_stopping = True  # Enable early stopping
    patience = 100  # Define patience for early stopping
    epochs_no_improve = 0  # Counter for early stopping
    best_val_loss = float('inf')  # Set initial loss to infinity

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

        if early_stopping:
            # Check for improvement
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0  # Reset counter
            else:
                epochs_no_improve += 1  # Increment if no improvement
            
            # Early stopping condition
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_val_loss:.4f}")
                break

    return train_losses, val_losses



# Train the model ------------------------------------------------------------
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



# Evaluate the model on the test set -----------------------------------------
def evaluate_model(model, test_X, test_y, criterion):
    model.eval()
    with torch.no_grad():
        test_output = model(test_X)
        test_loss = criterion(test_output, test_y).item()
    return test_loss

# Compute test loss after training
test_loss = evaluate_model(model, test_X, test_y, criterion)
print(f"Final Test Loss: {test_loss:.4f}")



# Inverse transform the predictions for graphical evaluation ------------------------------------------
# Function to inverse transform the normalized values back to original price scale
def inverse_transform(preds, min_val, max_val):
    return (preds + 1) * (max_val - min_val) / 2 + min_val

# Get predictions on test set
model.eval()
predictions = []
actual_values = []

with torch.no_grad():
    for i in range(len(test_X)):
        x_test = test_X[i].unsqueeze(0)  # Add batch dimension
        y_test = test_y[i].unsqueeze(0)  # Add batch dimension
        pred = model(x_test)
        predictions.append(pred.item())
        actual_values.append(y_test.item())

# Convert back to original price scale
predictions = inverse_transform(np.array(predictions), target_min, target_max)
actual_values = inverse_transform(np.array(actual_values), target_min, target_max)

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label="Actual Price", color='blue', linewidth=2)
plt.plot(predictions, label="Predicted Price", color='red', linestyle='dashed', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted Price (Test Set)")
plt.legend()
plt.grid(True)
plt.show()