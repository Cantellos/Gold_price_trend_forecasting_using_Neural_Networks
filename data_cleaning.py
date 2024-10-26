import pandas as pd
from pathlib import Path

# Define the relative path to the data file containing the corpus of the files to be analyzed
csv_file_path = (Path(__file__).resolve().parent / 'dataset').as_posix()

# Load the data from the CSV file
df = pd.read_csv(csv_file_path + '/XAU_1w_data_2004_to_2024-09-20.csv')
print(df)

# Remove the rows with missing values
df_clean = df.dropna()

if len(df) == len(df_clean):
    print("No missing values found")
else:
    print(f"Missing values found: {len(df) - len(df_clean)} rows removed") 
