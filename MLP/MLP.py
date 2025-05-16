import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 0. Loading and Normalizing the Dataset =====
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', 'RSI']
target = 'future_close'


# Split the dataset into 70% training, 15% validation, 15% testing
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
training = data[:train_size]
validation = data[train_size:train_size + val_size]
testing = data[train_size + val_size:]

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(training[features])
train_data = scaler.transform(training[features])
val_data = scaler.transform(validation[features])
test_data = scaler.transform(testing[features])

train_target = training[[target]].values
val_target = validation[[target]].values
test_target = testing[[target]].values

def create_tensor_dataset(data, target):
    x = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    return TensorDataset(x, y)

train_dataset = create_tensor_dataset(train_data, train_target)
val_dataset = create_tensor_dataset(val_data, val_target)
test_dataset = create_tensor_dataset(test_data, test_target)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
lr = 0.001

model = FullyConnected(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print("✅ New best val_loss. Model weights saved in memory.")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1}.")
                break

    if best_model_state is not None:
        model_path = (Path(__file__).resolve().parent.parent / 'models' / 'MLP_model.pth').as_posix()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, model_path)
        print("📁 Best model weights saved to disk.")

    return train_losses, val_losses

# Start training
num_epochs = 300
patience = 50
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience)


# ===== Plotting the Losses =====
starting_epoch = 30  # Start plotting from this epoch for graphic reasons
plt.figure(figsize=(12,6))
plt.plot(range(starting_epoch, len(train_losses) + 1), train_losses[starting_epoch-1:], label='Train Loss', marker='o')
plt.plot(range(starting_epoch, len(val_losses) + 1), val_losses[starting_epoch-1:], label='Validation Loss', marker='s')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss (excluding first {starting_epoch} epochs for graphic reasons)')
plt.show()


# ===== Testing the Model =====
# Load the best model weights
model_path = (Path(__file__).resolve().parent.parent / 'models' / 'MLP_model.pth').as_posix()
model.load_state_dict(torch.load(model_path))

# Evaluate the model on the test set
model.eval()
predictions = []
actuals = []
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_loader:
        output = model(xb).squeeze()  # squeeze to remove extra dimension
        yb = yb.squeeze()             # and have them as a 1D tensor already
        test_loss += criterion(output, yb).item()
        predictions.extend(output.tolist())
        actuals.extend(yb.tolist())

test_loss /= len(test_loader)
print(f'\n📊 MSE Loss - Test set (MLP): {test_loss:.6f}')


# ===== Accuracy-based Loss Calculation =====
def accuracy_based_loss(predictions, targets, threshold):
    corrects = 0
    for i in range(len(predictions)):
        if abs(predictions[i] - targets[i]) <= threshold/100 * targets[i]:
            corrects += 1
    accuracy = corrects / len(predictions)
    print(f"Correct predictions: {corrects}, Total predictions: {len(predictions)}")
    return accuracy

threshold = 1 # % threshold for accuracy
accuracy = accuracy_based_loss(predictions, actuals, threshold)
print(f'\nAccuracy - Test set (MLP): {accuracy*100:.4f}% of correct predictions within {threshold}%')


# ===== Plotting Predictions vs Actuals values =====
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()