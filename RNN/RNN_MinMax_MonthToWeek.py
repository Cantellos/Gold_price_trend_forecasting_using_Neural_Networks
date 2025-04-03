import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load and preprocess the dataset --------------------------------------------
# Load the dataset
file_path = (Path(__file__).resolve().parent.parent / '.data' / 'dataset' / 'XAU_1d_data_2004_to_2024-09-20.csv').as_posix()
data = pd.read_csv(file_path)

# Separate features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'EMA_12-26', 'EMA_50-200', '%K', '%D', 'RSI']
target = 'future_close'

# Normalizzazione delle feature
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data)

# Creazione delle sequenze di input (30 giorni) e output (7 giorni)
sequence_length = 30
forecast_horizon = 7

def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data['Close'].iloc[i+seq_length:i+seq_length+horizon].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data, sequence_length, forecast_horizon)

# Divisione in train, validation e test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Conversione in tensori
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Definizione del modello
class GRUForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUForecast, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Usa l'ultimo time step
        return out

# Istanziamento del modello
input_size = X_train.shape[2]  # Numero di feature
hidden_size = 64  # Unità nascoste
num_layers = 2
output_size = forecast_horizon  # 7 giorni di output

model = GRUForecast(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 50
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        val_losses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Valutazione sul test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Visualizzazione dei risultati
plt.figure(figsize=(12, 6))
plt.plot(y_test.numpy().flatten(), label='Actual', color='blue')
plt.plot(y_pred.numpy().flatten(), label='Predicted', color='red', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.legend()
plt.show()