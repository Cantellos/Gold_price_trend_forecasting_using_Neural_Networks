import pandas as pd
from pathlib import Path

dataset_dir = Path(__file__).parent / 'dataset'
files = list(dataset_dir.glob('*.csv'))

for file in files:
    # Load the data from the CSV file
    df = pd.read_csv(file)

    # Drop useless columns (Date and Time), add Target variable (shifted because it has to be the future price, not current) and drop NaN values
    df = df.drop(['Date', 'Time'], axis=1)
    df['future_close'] = df['Close'].shift(-1)

    # Drop the rows with missing values (some indicators require a certain number of previous values)
    df_clean = df.dropna()

    # Save the cleaned data to a new CSV file
    df_clean.to_csv(file, index=False)