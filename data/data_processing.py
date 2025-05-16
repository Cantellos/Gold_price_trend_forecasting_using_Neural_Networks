import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def load_and_process_data(filename):
    # ===== Loading, Processing and Normalizing the Dataset =====
    file_path = (Path(__file__).resolve().parent.parent / 'data' / 'dataset' / filename).as_posix()
    data = pd.read_csv(file_path)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'EMA_50', 'EMA_200', 'RSI']
    #features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'EMA_12-26', 'EMA_50-200', 'RSI']
    target = 'future_close'

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
    scaler = MinMaxScaler()
    scaler.fit(training[features])

    train_data = scaler.transform(training[features])
    val_data = scaler.transform(validation[features])
    test_data = scaler.transform(testing[features])

    train_target = training[[target]].values
    val_target = validation[[target]].values
    test_target = testing[[target]].values

    # Create TensorDataset
    def create_tensor_dataset(data, target):
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        return TensorDataset(x, y)

    train_dataset = create_tensor_dataset(train_data, train_target)
    val_dataset = create_tensor_dataset(val_data, val_target)
    test_dataset = create_tensor_dataset(test_data, test_target)

    # Create DataLoader for each dataset
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, features, target