import pandas as pd
from pathlib import Path


#Define which time frame to analyze
file_name = '/XAU_1w_data_2004_to_2024-09-20.csv'

# Define the relative path to the data file containing the corpus of the files to be analyzed
csv_file_path = (Path(__file__).resolve().parent / 'dataset').as_posix()

# Load the data from the CSV file
df = pd.read_csv(csv_file_path + file_name)

# Remove the rows with missing values
df_clean = df.dropna()
if len(df) == len(df_clean):
    print("No missing values found")
else:
    print(f"Missing values found: {len(df) - len(df_clean)} rows removed")   


# Adding financial indicators to the dataset

# MA (Moving Average) 
df_clean['MA_200'] = df_clean['Close'].ewm(span=200, adjust=False).mean()

# EMA (Exponential Moving Average)
df_clean['EMA_12-26'] = (df_clean['Close'].ewm(span=12, adjust=False).mean()) - (df_clean['Close'].ewm(span=26, adjust=False).mean())
df_clean['EMA_50-200'] = (df_clean['Close'].ewm(span=50, adjust=False).mean()) - (df_clean['Close'].ewm(span=200, adjust=False).mean())

#Stocastic Oscillator: measures the location of the close relative to the high-low range over a set period of time
lowest_low = df_clean['Low'].rolling(window=14).min()
highest_high = df_clean['High'].rolling(window=14).max()
# Calcola %K
k_line = 100 * ((df_clean['Close'] - lowest_low) / (highest_high - lowest_low))
# Calcola %D come media mobile semplice di %K
d_line = k_line.rolling(window=3).mean()
df_clean['%K']= k_line
df_clean['%D']= d_line

# RSI (Relative Strength Index): measures the speed and change of price movements
# Calcola le variazioni giornaliere
delta = df_clean['Close'].diff(1)

# Calcola i guadagni e le perdite
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# Media mobile esponenziale dei guadagni e delle perdite
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()

# Calcola l'RS
rs = avg_gain / avg_loss

# Calcola l'RSI
df_clean['RSI'] = 100 - (100 / (1 + rs))

#TODO: Add Fibonacci retracement levels

# Drop the rows with missing values (since some indicatore require a certain number of previous values)
df_clean = df_clean.dropna()

# Save the cleaned data with financial indicators to a new CSV file
df_clean.to_csv(csv_file_path + '/XAU_1w_data_2004_to_2024-09-20_cleaned.csv', index=False)

print(df_clean)
