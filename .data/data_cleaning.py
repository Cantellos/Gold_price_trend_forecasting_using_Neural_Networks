import pandas as pd
from pathlib import Path

dataset_dir = Path(__file__).parent / 'dataset'
files = list(dataset_dir.glob('*.csv'))

# --- Data Cleaning ---
for file in files:
    # Load the data from the CSV file
    df = pd.read_csv(file)

    # Divide the single column into 6 columns by splitting on ';'
    if len(df.columns) == 1:
        columns = df.columns[0].split(';')
        df = df[df.columns[0]].str.split(';', expand=True)
        df.columns = columns

    # Drop useless Date column
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)

    # Drop the rows with missing values (some indicators require a certain number of previous values)
    df_clean = df.dropna()

    # Save the cleaned data to a new CSV file
    df_clean.to_csv(file, index=False)