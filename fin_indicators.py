import pandas as pd
from pathlib import Path
from data_cleaning import df_clean as df

# --- Adding financial indicators to the dataset ---

# MA (Moving Average) 
df['MA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# EMA (Exponential Moving Average)
df['EMA_12-26'] = (df['Close'].ewm(span=12, adjust=False).mean()) - (df['Close'].ewm(span=26, adjust=False).mean())
df['EMA_50-200'] = (df['Close'].ewm(span=50, adjust=False).mean()) - (df['Close'].ewm(span=200, adjust=False).mean())

#Stocastic Oscillator: measures the location of the close relative to the high-low range over a set period of time
lowest_low = df['Low'].rolling(window=14).min()
highest_high = df['High'].rolling(window=14).max()
k_line = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
d_line = k_line.rolling(window=3).mean()
df['%K']= k_line
df['%D']= d_line

# RSI (Relative Strength Index): measures the speed and change of price movements
delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
# EMA of gains and losses
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()
# Calculate the Relative Strength
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

#TODO: Add Fibonacci retracement levels
#TODO: Add Bollinger bands retracement levels
 #TODO: check if everything wokrs
# Adding FED interest rates and inflation rates (stock prices are influenced by these factors)
# Load the dataset and specify columns to load
fed_path = (Path(__file__).resolve().parent / 'dataset' / '1Month_interest_unemployment,inflation').as_posix()
fed = pd.read_csv(fed_path, usecols=['Year', 'Month', 'Effective Federal Funds Rate','Unemployment Rate','Inflation Rate'])
# Select rows that match the time frame of our stock data
fed = fed[fed['Year'] > 2003]
fed = fed.dropna()
#fed.to_csv(fed_path, index=False)
print(fed)