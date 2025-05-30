import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def load_and_process_data(filename, seq_len, pred_len, batch_size):
    # ===== Loading, Processing and Normalizing the Dataset =====
    file_path = (Path(__file__).resolve().parent.parent / 'data' / 'dataset' / filename).as_posix()
    data = pd.read_csv(file_path)

    # All available features: Open, High, Low, Close, Volume, MA_50, MA_200, EMA_12, EMA_26, EMA_12-26, EMA_50, EMA_200, EMA_50-200, %K, %D, RSI
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'EMA_12-26', 'EMA_50-200', 'EMA_200', 'RSI']
    target = ['future_close']   

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)


    # Split the dataset into 70% training, 15% validation, 15% testing
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)

    training = data.iloc[:train_size].copy()
    validation = data.iloc[train_size:train_size+val_size].copy()
    testing = data.iloc[train_size+val_size:].copy()

    # Add target variable columns
    training['future_close'] = training['Close'].shift(-1)
    validation['future_close'] = validation['Close'].shift(-1)
    testing['future_close'] = testing['Close'].shift(-1)

    # Drop rows with NaNs due to shifting
    training.dropna(inplace=True)
    validation.dropna(inplace=True)
    testing.dropna(inplace=True)

    # Reset indices
    training.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)
    testing.reset_index(drop=True, inplace=True)


    # Normalize the features using MinMaxScaler
    features_scaler = MinMaxScaler()
    features_scaler.fit(training[features])

    train_data = features_scaler.transform(training[features])
    val_data = features_scaler.transform(validation[features])
    test_data = features_scaler.transform(testing[features])

    target_scaler = MinMaxScaler()
    target_scaler.fit(training[target])

    train_target = target_scaler.transform(training[target])
    val_target = target_scaler.transform(validation[target])
    test_target = target_scaler.transform(testing[target])


    # Sliding windows to create sequences
    def create_sequences(data, target, seq_len, pred_len):
        sequences = []
        targets = []
        for i in range(len(data) - seq_len - pred_len + 1):
            seq = data[i:i + seq_len]
            label = target[i + seq_len:i + seq_len + pred_len]
            sequences.append(seq)
            targets.append(label)
        return np.array(sequences), np.array(targets)
    
    train_data, train_target = create_sequences(train_data, train_target, seq_len, pred_len)
    val_data, val_target = create_sequences(val_data, val_target, seq_len, pred_len)
    test_data, test_target = create_sequences(test_data, test_target, seq_len, pred_len)


    # Create TensorDataset
    def create_tensor_dataset(data, target):
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        return TensorDataset(x, y)

    train_dataset = create_tensor_dataset(train_data, train_target)
    val_dataset = create_tensor_dataset(val_data, val_target)
    test_dataset = create_tensor_dataset(test_data, test_target)


    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, features, pred_len, features_scaler, target_scaler