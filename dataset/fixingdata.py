import pandas as pd

# Conta i valori mancanti per ogni colonna
missing_values = df.isnull().sum()
# Visualizza le righe con valori mancanti
missing_data = df[df.isnull().any(axis=1)]
