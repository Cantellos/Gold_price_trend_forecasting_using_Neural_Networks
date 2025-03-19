import pandas as pd
from pathlib import Path

# Adding FED interest rates and US unemployment and inflation rates (stock prices are influenced by these factors)

# Load the dataset and specify columns to load
fed_path = (Path(__file__).resolve().parent / 'FED_1Month_interest,unemployment,inflation.csv').as_posix()
fed = pd.read_csv(fed_path, usecols=['Year', 'Month', 'Interest','Unemployment','Inflation'])

# Drop rows with missing values and save the new file
fed = fed.dropna()
fed.to_csv(fed_path, index=False)