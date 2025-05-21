import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.RNN_data_processing import load_and_process_data

# Define the type of forecasting
seq_len = 30     # Length of the input sequence
pred_len = 7      # Length of the prediction sequence
batch_size = 128   # Batch size for training

# ===== Loading, Processing and Normalizing the Dataset =====
train_loader, val_loader, test_loader, features, target = load_and_process_data('XAU_1d_data.csv', seq_len, pred_len, batch_size)


# ===== Building the RNN Model =====
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)       
        out = out[:, -1, :]            
        out = self.fc(out)             
        out = out.unsqueeze(-1)    # (batch_size, 1, 1) to match yb shape       
        return out

input_size = len(features)
hidden_size = 64
num_layers = 1
output_size = pred_len
lr = 0.001

model = RNN(input_size, hidden_size, num_layers, output_size)
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
#optimizer = optim.RMSprop(model.parameters(), lr)
optimizer = optim.Adam(model.parameters(), lr)

# ===== Training the Model =====
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
            loss = criterion(output, yb) # Flatten to avoid implicit broadcasting errors
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
                loss = criterion(output.view(-1), yb.view(-1)) # Flatten to avoid implicit broadcasting errors
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
        model_path = (Path(__file__).resolve().parent.parent / 'models' / 'RNN2_model.pth').as_posix()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, model_path)
        print("Best model weights saved to disk.")

    return train_losses, val_losses

# Start training
num_epochs = 300
patience = 30
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience)


# ===== Plotting the Losses =====
starting_epoch = 10  # Start plotting from this epoch for graphic reasons
plt.figure(figsize=(12,6))
plt.plot(range(starting_epoch, len(train_losses) + 1), train_losses[starting_epoch-1:], label='Train Loss', marker='o')
plt.plot(range(starting_epoch, len(val_losses) + 1), val_losses[starting_epoch-1:], label='Validation Loss', marker='s')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss (excluding first {starting_epoch} epochs for graphic reasons) - RNN: Multi-Step Prediction')
plt.show()


# ===== Testing the Model =====
# Load the best model weights
model_path = (Path(__file__).resolve().parent.parent / 'models' / 'RNN2_model.pth').as_posix()
model.load_state_dict(torch.load(model_path, weights_only=False))

# Evaluate the model on the test set
model.eval()
predictions = []
actuals = []
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_loader:
        output = model(xb).squeeze(-1)   
        yb = yb.squeeze(-1)              
        test_loss += criterion(output, yb).item()
        predictions.extend(output.tolist())
        actuals.extend(yb.tolist())

test_loss /= len(test_loader)
print(f'\nMSE Loss - Test set (RNN: Multi-Step Prediction): {test_loss:.6f}')


# ===== Accuracy-based Loss Calculation =====
def accuracy_based_loss(predictions, targets, threshold):
    flat_preds = [item for sublist in predictions for item in sublist]
    flat_targets = [item for sublist in targets for item in sublist]

    corrects = 0
    for i in range(len(flat_preds)):
        if abs(flat_preds[i] - flat_targets[i]) <= threshold/100 * abs(flat_targets[i]):
            corrects += 1
    accuracy = corrects / len(flat_preds)
    print(f"Correct predictions: {corrects}, Total predictions: {len(flat_preds)}")
    return accuracy

threshold = 1 # % threshold for accuracy
accuracy = accuracy_based_loss(predictions, actuals, threshold)
print(f'\nAccuracy - Test set (RNN: Multi-Step Prediction): {accuracy*100:.4f}% of correct predictions within {threshold}%')


# ===== Average Percentage % Error Calculation =====
def average_percentage_error(predictions, actuals):
    flat_preds = np.array([item for sublist in predictions for item in sublist])
    flat_actuals = np.array([item for sublist in actuals for item in sublist])

    percent_errors = np.abs((flat_preds - flat_actuals) / flat_actuals) * 100
    avg_percent_error = np.mean(percent_errors)
    return avg_percent_error

percentage_error = average_percentage_error(predictions, actuals)
print(f'\nAverage % Error - Test set (RNN: Multi-Step Prediction): {percentage_error:.4f}% of average error')


# ===== Plotting Multi-Step Forecasting with Error Bars =====
def plot_actual_vs_mean_predicted_with_error(actuals, predicted, pred_len):
    actuals = np.array(actuals)
    N = len(actuals)

    # Lists to accumulate predictions per time step
    pred_dict = {i: [] for i in range(N)}

    # Aggregate predictions
    for i, pred_seq in enumerate(predicted):
        for j, value in enumerate(pred_seq):
            idx = i + j
            if idx < N:
                pred_dict[idx].append(value)

    # Compute mean and std
    mean_predictions = []
    std_predictions = []
    valid_indices = []

    for i in range(N):
        preds = pred_dict[i]
        if preds:
            mean_predictions.append(np.mean(preds))
            std_predictions.append(np.std(preds))
            valid_indices.append(i)

    x = np.array(valid_indices)
    y_true = actuals[x]
    y_pred = np.array(mean_predictions)
    y_std = np.array(std_predictions)

    # Plot with error bands
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_true, label='Actual', color='blue')
    plt.plot(x, y_pred, label='Mean Predicted', color='orange')
    plt.fill_between(x, y_pred - y_std, y_pred + y_std, color='orange', alpha=0.3, label='±1 STD')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Mean Predicted Prices (with Error Bands)')
    plt.legend()
    plt.grid(True)
    plt.show()
# Call the function to plot
plot_actual_vs_mean_predicted_with_error(actuals, predictions, pred_len)