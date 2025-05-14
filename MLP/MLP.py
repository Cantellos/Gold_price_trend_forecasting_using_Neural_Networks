import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# TODO: finding loss + optimizer combo with best performance


# ===== 0. Loading and Normalizing the Dataset =====
# Load the dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Choose features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Split dataset (70% train, 15% val, 15% test)
train_size = int(len(data) * 0.7)   
val_size = int(len(data) * 0.15)

training = data[:train_size]
validation = data[train_size:train_size + val_size]
testing = data[train_size + val_size:]


# ===== 1. Normalization =====
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
    x = torch.tensor(data, dtype=torch.float32)  # Add dimension to ensure the correct shape for RNN input
    y = torch.tensor(target, dtype=torch.float32)
    return x, y

train_x, train_y = create_tensor_dataset(train_data, train_target)
val_x, val_y = create_tensor_dataset(val_data, val_target)
test_x, test_y = create_tensor_dataset(test_data, test_target)


# ===== 2. Definition of the MLP (Fully Connected Layer) =====
class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = len(features)
hidden_size = 64
output_size = 1
lr=0.001

model = FullyConnected(input_size, hidden_size, output_size)

# criterion = nn.SmoothL1Loss()
criterion = nn.MSELoss()

# optimizer = optim.Adam(model.parameters(), lr)
optimizer = optim.RMSprop(model.parameters(), lr)

# ===== 3. Training Function =====
def train_model(model, train_x, train_y, val_x, val_y, criterion, optimizer, num_epochs, patience):
    train_losses = []
    val_losses = []
    patience = patience  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):

        # Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_losses.append(train_loss)  

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
            val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # salva in RAM
            print("✅ New best val_loss. Model weights saved in memory.")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(F"⛔ Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs..")
                break
    
    # At the end save the best model to disk
    if best_model_state is not None:
        model_path = (Path(__file__).resolve().parent.parent / 'models' / 'MLP2_model.pth').as_posix()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(best_model_state, model_path)
        print("📁 Best model weights saved to disk.")

    return train_losses, val_losses


# ===== 4. Training the Model =====
num_epochs = 500
patience = 100
train_losses, val_losses = train_model(model, train_x, train_y, val_x, val_y, criterion, optimizer, num_epochs, patience)


# ===== 5. Plotting the Losses =====
starting_epoch = 50  # Start plotting from epoch 50
plt.figure(figsize=(11,6))
plt.plot(range(starting_epoch, len(train_losses) + 1), train_losses[starting_epoch-1:], label='Train Loss', marker='o')
plt.plot(range(starting_epoch, len(val_losses) + 1), val_losses[starting_epoch-1:], label='Validation Loss', marker='s')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (excluding first 2 for graphic reasons)')
plt.show()


# ===== 6. Testing the Model =====
model_path = (Path(__file__).resolve().parent.parent / 'models' / 'MLP2_model.pth').as_posix()
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    test_outputs = model(test_x)
    test_loss = criterion(test_outputs, test_y).item()
    predictions = test_outputs.tolist()
print(f'\n📊 MSE Loss - Test set (MLP): {test_loss:.6f}')


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
    print(f"Correct predictions: {corrects}, Total predictions: {len(predictions)}")
    return accuracy

threshold = 0.02  # 5% tolerance
predictions = scaler.inverse_transform(predictions)  # Inverse transform to get actual prices
actuals = scaler.inverse_transform(test_y)  # Inverse transform to get actual prices
loss = accuracy_based_loss(predictions, actuals, threshold=threshold)
print(f'\nAccuracy - Test set (MLP): {loss*100:.4f}% of correct predictions within {threshold*100}%\n')


# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()