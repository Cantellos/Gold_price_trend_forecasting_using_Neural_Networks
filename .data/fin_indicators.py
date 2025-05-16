from data_cleaning import files
import pandas as pd

# --- Adding financial indicators to the dataset ---
for file in files:
    # Load the data from the CSV file
    data = pd.read_csv(file)

    # CALCULATE FINANCIAL INDICATORS
    # MA: Moving Average
    data['MA_200'] = data['Close'].rolling(window=200).mean()

    # EMA: Exponential Moving Average
    # EMA_12-26: 12-day EMA - 26-day EMA
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['EMA_12-26'] = data['EMA_12'] - data['EMA_26']
    # EMA 50-200
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['EMA_50-200'] = data['EMA_50'] - data['EMA_200']

    # SO: Stochastic Oscillator
    lowest_low = data['Low'].rolling(window=14).min()
    highest_high = data['High'].rolling(window=14).max()
    k_line = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d_line = k_line.rolling(window=3).mean()
    data['%K'] = k_line
    data['%D'] = d_line   
    # TODO: Change the Stochastic Oscillator in 1 unique value

    # RSI: Relative Strength Index
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Add Target variable (shifted because it's the future price, not current)
    data['future_close'] = data['Close'].shift(-1)

    # Save the data with the financial indicators to a new CSV file without rows with missing values
    data = data.dropna()
    data.to_csv(file, index=False)

#TODO: Add Fibonacci retracement levels
#TODO: Add Bollinger bands retracement levels