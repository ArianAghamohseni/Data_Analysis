import pandas as pd
import yfinance as yf



gold_data = yf.download('GLD', start='2010-01-01', end='2020-12-31')
gas_data = yf.download('NG=F', start='2010-01-01', end='2020-12-31')





# Data preprocessing (using pandas):



# Data sampling and alignment
# Resample the data to a specific time frequency, such as monthly or weekly
gold_data = gold_data.resample('M').last()
gas_data = gas_data.resample('M').last()



# Handling null data
# Fill any missing values in the data with appropriate methods
gold_data = gold_data.fillna(method='ffill')
gas_data = gas_data.fillna(method='bfill')



# Data normalization
# Normalize the data using a suitable normalization technique, such as Min-Max scaling or Z-score normalization
gold_data_normalized = (gold_data - gold_data.min()) / (gold_data.max() - gold_data.min())
gas_data_normalized = (gas_data - gas_data.min()) / (gas_data.max() - gas_data.min())



# Convert the data into a stationary series by differencing
gold_data_stationary = gold_data_normalized.diff().dropna()
gas_data_stationary = gas_data_normalized.diff().dropna()





# Save the modified data to CSV files
gold_data_stationary.to_csv('D:\Sharif University of Tech\Term 4\AP\Project\Third faze\CSV Files\gold_data_stationary.csv')
gas_data_stationary.to_csv('D:\Sharif University of Tech\Term 4\AP\Project\Third faze\CSV Files\gas_data_stationary.csv')
