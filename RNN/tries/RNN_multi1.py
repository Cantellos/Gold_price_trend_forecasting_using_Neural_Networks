import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.MLP_data_processing import load_and_process_data

# Define the type of forecasting
seq_len=30      # Length of the input sequence
pred_len=7      # Length of the prediction sequence (it supports both single-step and multi-step forecasting)

# ===== Loading, Processing and Normalizing the Dataset =====
train_loader, val_loader, test_loader, features, target = load_and_process_data('XAU_1d_data.csv', seq_len, pred_len)


# ===== Building the RNN Model =====
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]        # Take last time step
        out = self.fc(out)         # Map to output_size (1 or 7)
        return out

input_size = len(features)
hidden_size = 64
num_layers = 1
output_size = pred_len
lr = 0.001

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()
optimizer = optim.RMSprop(model.parameters(), lr)
# optimizer = optim.Adam(model.parameters(), lr)


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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print("New best val_loss. Model weights saved in memory.")
        else:
            epochs_no_improve += 1
            print(f"No improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
    
    # Model checkpointing             
    if best_model_state is not None:
        model_path = (Path(__file__).resolve().parent.parent / 'models' / 'MLP_model.pth').as_posix()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, model_path)
        print("Best model weights saved to disk.")

    return train_losses, val_losses

# Start training
num_epochs = 200
patience = 30
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
model.load_state_dict(torch.load(model_path, weights_only=False))

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
print(f'\nMSE Loss - Test set (MLP): {test_loss:.6f}')


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