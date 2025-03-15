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
features = ['Open', 'High', 'Low', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
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



# Convert data to PyTorch tensors --------------------------------------------
def create_tensor_dataset(df, features, target):
    X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
    y = torch.tensor(df[target].values, dtype=torch.float32).unsqueeze(1)  # Make sure y has correct shape
    return X, y

train_X, train_y = create_tensor_dataset(train_data, features, target)
val_X, val_y = create_tensor_dataset(val_data, features, target)
test_X, test_y = create_tensor_dataset(test_data, features, target)



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



# Set hyperparameters and instantiate the model ------------------------------
input_size = train_X.shape[2] # Exclude the target variable
hidden_size = 64
num_layers = 5
num_epochs=500
output_size = 1
lr=0.001

# Instantiate the model, define loss function and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)



# Define the training function ------------------------------------------------
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



# Train the model -------------------------------------------------------------
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



# Evaluate the model on the test set ------------------------------------------
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