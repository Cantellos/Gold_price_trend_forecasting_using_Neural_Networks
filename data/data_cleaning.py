import pandas as pd
from pathlib import Path

#Define which time frame to analyze
file_name = '/XAU_1w_data_2004_to_2024-09-20.csv'

# Define the relative path to the data file containing the corpus of the files to be analyzed
csv_file_path = (Path(__file__).resolve().parent / 'dataset').as_posix()

file_path = csv_file_path + file_name

# Load the data from the CSV file
df = pd.read_csv(file_path)

# Remove the rows with missing values
df_clean = df.dropna()
if len(df) == len(df_clean):
    print("No missing values found")
else:
    print(f"Missing values found: {len(df) - len(df_clean)} rows removed")   

# Drop the rows with missing values (some indicators require a certain number of previous values)
df_clean = df_clean.dropna()
# Save the cleaned data with financial indicators to a new CSV file
df_clean.to_csv(csv_file_path + '/XAU_1w_data_2004_to_2024-09-20.csv', index=False)