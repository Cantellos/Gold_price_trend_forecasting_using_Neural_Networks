from data_cleaning import files
import pandas as pd

# --- Adding financial indicators to the dataset ---
for file in files:
    # Load the data from the CSV file
    df = pd.read_csv(file)

    # Remove the rows with missing values
    df_clean = df.dropna()

    # CALCULATE FINANCIAL INDICATORS
    # MA: Moving Average
    df_clean['MA_200'] = df_clean['Close'].ewm(span=200, adjust=False).mean()

    # EMA: Exponential Moving Average
    df_clean['EMA_12-26'] = (df_clean['Close'].ewm(span=12, adjust=False).mean()) - (df_clean['Close'].ewm(span=26, adjust=False).mean())
    df_clean['EMA_50-200'] = (df_clean['Close'].ewm(span=50, adjust=False).mean()) - (df_clean['Close'].ewm(span=200, adjust=False).mean())

    # SO: Stochastic Oscillator
    lowest_low = df_clean['Low'].rolling(window=14).min()
    highest_high = df_clean['High'].rolling(window=14).max()
    k_line = 100 * ((df_clean['Close'] - lowest_low) / (highest_high - lowest_low))
    d_line = k_line.rolling(window=3).mean()
    df_clean['%K'] = k_line
    df_clean['%D'] = d_line
# TODO: Fix the Stochastic Oscillator in 1 unique value

    # Relative Strength Index
    delta = df_clean['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df_clean['RSI'] = 100 - (100 / (1 + rs))

    # Save the data with the financial indicators to a new CSV file without rows with missing values
    df = df.dropna()
    df_clean.to_csv(file, index=False)

#TODO: Add Fibonacci retracement levels
#TODO: Add Bollinger bands retracement levels