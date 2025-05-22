This project goal is to build a neural network that predict the future price trend of the XAU/USD pair. I worked on the free dataset available at the following link: https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024 and I also used other data found online regarding FED rates.

How to use:
- Download the dataset at the link above
- Move into the 'data' directory'
- Create here a new 'dataset' directory
- Put the dowloaded dataset into the new relevant directory
- Run the file data_cleaning.py
- Run the file fin_indicators.py
OPTIONAL: - Run the file dataset_visualisation.py to visualize price trend and some financial indicators


Now that's all set up, you can choose whatever model and type of prediction you want to use and run the relative file:
SINGLE-step prediction
  - MLP1.py uses a MLP with 1 hidden layer
  - MLP2.py uses a MLP with 2 hidden layers
  - RNN1.py uses a RNN, can take multiple-step input
MULTI-step prediction
  - RNN2.py uses a RNN, should take multiple-step input in order to make a (smaller) multi-step prediction

All the files are currenty set up in order to work with a 1-Day time frame, using the file 'XAU_1d_data.csv'. If you want to use a different time frame you need to change the name of the file in the first rows of the program.

NB: if the names of the downloaded files had changed, you have to change the csv file name you want to work on in order to match them.
