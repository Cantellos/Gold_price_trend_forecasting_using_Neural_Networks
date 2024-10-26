#open the dataset C:/Users/andre/Documents/GitHub/Lab-DL/dataset/XAU_1w_data_2004_to_2024-09-20.csv and clean it
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = os.path.join('..', 'dataset', 'XAU_1w_data_2004_to_2024-09-20.csv')
df = pd.read_csv(path)