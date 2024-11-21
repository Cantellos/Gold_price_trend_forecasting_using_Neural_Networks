import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM,GRU,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


dataset_dir = Path(__file__).parent / '.data' / 'dataset'
files = list(dataset_dir.glob('*.csv'))

for file in files:
    if file.name == 'XAU_1d_data_2004_to_2024-09-20.csv':
        data = pd.read_csv(file)
       
# Aggiungi la colonna del target (prezzo di chiusura futura)
data['future_close'] = data['Close'].shift(-1)       

#Rimuovi le colonne 'Data' and 'Time'
data = data.drop(columns=['Date','Time'])
#FOR NOW: Rimuovi le colonne '%K' and '%D'
data = data.drop(columns=['%K','%D'])
# TODO: HANDLE STOCHASTIC OSCILLATOR (UNIQUE VALUE; DIVERGENCES and CROSSOVER)

# Numero totale di campioni
num_samples = len(data)

# Indici per dividere il dataset
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

# Divisione del dataset
train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size + val_size]
test_data = data.iloc[train_size + val_size:]

# Separazione delle feature e dei target per ogni set
train_features = torch.tensor(train_data.drop(columns=['future_close']).values, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_data['future_close'].values, dtype=torch.float32).unsqueeze(1)

val_features = torch.tensor(val_data.drop(columns=['future_close']).values, dtype=torch.float32)
val_labels = torch.tensor(val_data['future_close'].values, dtype=torch.float32).unsqueeze(1)

test_features = torch.tensor(test_data.drop(columns=['future_close']).values, dtype=torch.float32)
test_labels = torch.tensor(test_data['future_close'].values, dtype=torch.float32).unsqueeze(1)

# Normalizzazione delle feature
scaler = MinMaxScaler()
columns_to_normalize = ['Open','High','Low','Close','Volume','MA_200','EMA_12-26','EMA_50-200','RSI']
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# --- MODEL IMPLEMENTATION ---

class GRUPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUPricePredictor, self).__init__()
        # Definisce il layer GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Layer fully connected per l'output finale
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Inizializza l'hidden state per il layer GRU
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        # Propaga l'input attraverso la GRU
        #TODO: questo X è un tensore di dimensione 2, ma la GRU vuole un tensore di dimensione 3
        out, _ = self.gru(x, h0)
        # Usa l'output dell'ultimo time step per fare la previsione finale
        out = self.fc(out[:, -1, :])  # Usa solo l'ultimo output della sequenza
        return out

# Parametri del modello
input_size = 9  # Numero di feature (open, high, low, close, volume)
hidden_size = 64  # Numero di unità nascoste nel layer GRU  #TODO: Try with 128
num_layers = 3  # Numero di layer GRU
output_size = 1  # Usiamo un singolo output per predire il prezzo

# Inizializza il modello
model = GRUPricePredictor(input_size, hidden_size, num_layers, output_size)

# --- ADDESTRAMENTO ---

# Definizione della loss function e dell'ottimizzatore
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500

# Liste per memorizzare la loss e le metriche durante l'addestramento
train_loss = []

for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()  # Resetta i gradienti
    outputs = model(train_features)
    loss = criterion(outputs, train_labels)  # Calcola la perdita
    loss.backward()  # Calcola i gradienti
    optimizer.step()  # Aggiorna i pesi
    
    # Aggiungi il valore della loss alla lista
    train_loss.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
