import pandas as pd
from pathlib import Path

dataset_dir = Path(__file__).parent / 'dataset'
files = list(dataset_dir.glob('*.csv'))

for file in files:
    # Load the data from the CSV file
    df = pd.read_csv(file)

    # Drop the rows with missing values (some indicators require a certain number of previous values)
    df_clean = df.dropna()

    # Save the cleaned data to a new CSV file
    df_clean.to_csv(file, index=False)