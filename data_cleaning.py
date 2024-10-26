import pandas as pd
from pathlib import Path

# Define the base directory (project root)
base_dir = Path(__file__).resolve().parent  # Goes up two levels from src/ to space/

# Define the relative path to the data file containing the corpus of the files to be analyzed
data_file_path = base_dir / 'dataset' / 'XAU_1w_data_2004_to_2024-09-20.csv'

# Convert the path to a string with forward slashes
input_directory_path = data_file_path.as_posix()

# Load the data from the CSV file
df = pd.read_csv(input_directory_path)
print(df)

# Count missing values for each column
missing_values = df.isnull().sum()

# Visualize the rows with missing values
missing_data = df[df.isnull().any(axis=1)]

# Remove the rows with missing values
df_clean = df.dropna()
print(df)

if len(df) == len(df_clean):
    print("No missing values found")
else:
    print(f"Missing values found: {len(df) - len(df_clean)} rows removed") 
