import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM,GRU,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

dataset_dir = Path(__file__).parent / '.data' / 'dataset'
files = list(dataset_dir.glob('*.csv'))

for file in files:
    if file.name == 'XAU_1Month_data_2004_to_2024-09-20.csv':
        df = pd.read_csv(file)
        
print(df)

"""
# Supponiamo che `data` sia il nostro DataFrame con tutte le feature e i target
data = pd.DataFrame()  # Sostituisci con il tuo dataset reale

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
train_features = train_data.drop(columns=['target'])  # Sostituisci 'target' con il nome della colonna target
train_labels = train_data['target']

val_features = val_data.drop(columns=['target'])
val_labels = val_data['target']

test_features = test_data.drop(columns=['target'])
test_labels = test_data['target']
"""

"""
df = files.drop(columns=['Time'])
dataset = np.array(df)
dataset.reshape(-1,1)
print(dataset)
"""
"""
plt.plot(dataset)

scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train=dataset[:train_size,:]
test=dataset[train_size:142,:]
def getdata(data,lookback):
    X,Y=[],[]
    for i in range(len(data)-lookback-1):
        X.append(data[i:i+lookback,0])
        Y.append(data[i+lookback,0])
    return np.array(X),np.array(Y).reshape(-1,1)
lookback=1
X_train,y_train=getdata(train,lookback)
X_test,y_test=getdata(test,lookback)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

model=Sequential()
model.add(LSTM(5,input_shape=(1,lookback)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
"""