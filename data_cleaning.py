import pandas as pd
from pathlib import Path

#TODO: import pandas

# Define the base directory (project root)
base_dir = Path(__file__).resolve().parent.parent  # Goes up two levels from src/ to space/

# Define the relative path to the data file containing the corpus of the files to be analyzed
data_file_path = base_dir / 'data' / 'XAU_1w_data_2004_to_2024-09-20.csv'

# Convert the path to a string with forward slashes
input_directory_path = data_file_path.as_posix()

# Load the data from the CSV file
df = pd.read_csv(input_directory_path)

# Conta i valori mancanti per ogni colonna
missing_values = df.isnull().sum()
# Visualizza le righe con valori mancanti
missing_data = df[df.isnull().any(axis=1)]
