import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def load_and_process_data(filename, seq_len, pred_len):
    # ===== Loading, Processing and Normalizing the Dataset =====
    file_path = (Path(__file__).resolve().parent.parent / 'data' / 'dataset' / filename).as_posix()
    data = pd.read_csv(file_path)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'EMA_12-26', 'EMA_50-200', 'EMA_200', 'RSI']
    
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Normalize features
    scaler = MinMaxScaler()
    scaler.fit(data[features])
    scaled_features = scaler.transform(data[features])
    
    close_values = data['Close'].values

    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(scaled_features[i:i+seq_len])  # [seq_len, input_size]
        y.append(close_values[i+seq_len:i+seq_len+pred_len])  # [pred_len]
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Adjust shape for single-step prediction
    if pred_len == 1:
        y = y.squeeze(-1)  # from [N, 1] → [N]

    # Split
    total = len(X)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_X, val_X, test_X = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    train_y, val_y, test_y = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    batch_size = 32
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, features, pred_len