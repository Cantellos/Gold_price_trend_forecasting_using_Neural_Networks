import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def load_and_process_data(filename, seq_len=10):
    file_path = (Path(__file__).resolve().parent.parent / 'data' / 'dataset' / filename).as_posix()
    data = pd.read_csv(file_path)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'EMA_12-26', 'EMA_50-200', 'EMA_200', 'RSI']
    target = 'future_close'

    # Create future target column
    data['future_close'] = data['Close'].shift(-1)

    # Drop final rows with NaNs
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Normalize features
    scaler = MinMaxScaler()
    scaler.fit(data[features])
    scaled_data = scaler.transform(data[features])
    target_data = data[target].values

    # Create sequences and targets
    def create_sequences(data, targets, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(targets[i+seq_len])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    X, y = create_sequences(scaled_data, target_data, seq_len)

    # Split into train, val, test
    total = len(X)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_X, val_X, test_X = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    train_y, val_y, test_y = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, features, target