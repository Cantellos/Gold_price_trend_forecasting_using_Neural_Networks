import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

# TODO aumenta numero di epoche

# ===== 1. Caricamento e Normalizzazione del Dataset =====
# Carica il dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Scegli le feature e il target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Normalizzazione feature per feature
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Split dataset (70% train, 15% val, 15% test)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

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
hidden_size = 128
output_size = 1

model = FullyConnected(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 3. Funzione di Training =====
def train_model(model, train_data, val_data, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

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
                x_val = torch.tensor(val_data[features].iloc[i].values, dtype=torch.float32).unsqueeze(0)
                y_val = torch.tensor(val_data[target].iloc[i], dtype=torch.float32).unsqueeze(0)

                output = model(x_val)
                loss = criterion(output.view(-1), y_val.view(-1))
                val_loss += loss.item()

            val_losses.append(val_loss / len(val_data))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')

    return train_losses, val_losses

# ===== 4. Addestramento del modello =====
num_epochs = 20
train_losses, val_losses = train_model(model, train_data, val_data, criterion, optimizer, num_epochs)

# ===== 5. Plot delle perdite =====
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
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
print(f'\nFinal Test Loss (MLP): {final_test_loss:.6f}')

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