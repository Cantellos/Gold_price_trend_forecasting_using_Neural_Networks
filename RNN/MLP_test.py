import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path


# Imposta il dispositivo per l'esecuzione su GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pd.options.mode.copy_on_write = True


# ===== 1. Caricamento e Normalizzazione del Dataset =====
# Carica il dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Scegli le feature e il target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Split dataset (70% train, 15% val, 15% test)
train_size = int(len(data) * 0.7)   
val_size = int(len(data) * 0.15)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Normalizzazione feature per feature
scaler = MinMaxScaler()

# Fit solo sul training set
scaler.fit(train_data[features])

# Trasforma training, validation e test usando lo stesso scaler
train_data[features] = scaler.transform(train_data[features])
val_data[features] = scaler.transform(val_data[features])
test_data[features] = scaler.transform(test_data[features])


# ===== 2. Definizione del MLP (Fully Connected Layer) =====
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
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr)


# ===== 3. Funzione di Training =====
def train_model(model, train_data, val_data, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    patience = 5  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i in range(len(train_data)):
            x_train = torch.tensor(train_data[features].iloc[i].values, dtype=torch.float32).unsqueeze(0)
            y_train = torch.tensor(train_data[target].iloc[i], dtype=torch.float32).unsqueeze(0)

            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output.view(-1), y_train.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_data))

        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(val_data)):
                #x_val = torch.tensor(val_data[features].iloc[i].values, dtype=torch.float32).unsqueeze(0)
                #y_val = torch.tensor(val_data[target].iloc[i], dtype=torch.float32).unsqueeze(0)
                x_val = torch.tensor(val_data[features].values, dtype=torch.float32)
                y_val = torch.tensor(val_data[target].values, dtype=torch.float32)  

                output = model(x_val)
                loss = criterion(output.view(-1), y_val.view(-1))
                val_loss += loss.item()

            val_losses.append(val_loss / len(val_data))

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


# ===== 4. Addestramento del modello =====
num_epochs = 50
train_losses, val_losses = train_model(model, train_data, val_data, criterion, optimizer, num_epochs)


# ===== 5. Plot delle perdite =====
plt.figure(figsize=(11,6))
plt.plot(range(3, len(train_losses) + 1), train_losses[2:], label='Train Loss', marker='o')
plt.plot(range(3, len(val_losses) + 1), val_losses[2:], label='Validation Loss', marker='s')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (excluding first 2 for graphic reasons)')
plt.show()


# ===== 6. Test finale sul test set =====
model.eval()
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for i in range(len(test_data)):
        x_test = torch.tensor(test_data[features].iloc[i].values, dtype=torch.float32).unsqueeze(0)
        y_test = torch.tensor(test_data[target].iloc[i], dtype=torch.float32).unsqueeze(0)

        output = model(x_test)
        loss = criterion(output.view(-1), y_test.view(-1))
        test_loss += loss.item()

        predictions.append(output.item())
        actuals.append(y_test.item())

final_test_loss = test_loss / len(test_data)
print(f'\nMSE Loss - Test set (MLP): {final_test_loss:.6f}')


# Accuracy Loss for Training
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
plt.plot(actuals, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title('Actual vs Predict Price (Test Set)')
plt.legend()
plt.grid(True)
plt.show()