import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Downloading and preprocessing data
gold_data = yf.download('GLD', start='2010-01-01', end='2020-12-31')
gas_data = yf.download('NG=F', start='2010-01-01', end='2020-12-31')

# Data sampling and alignment
gold_data = gold_data.resample('M').last()
gas_data = gas_data.resample('M').last()

# Handling null data
gold_data = gold_data.fillna(method='ffill')
gas_data = gas_data.fillna(method='bfill')

# Data normalization
gold_data_normalized = (gold_data - gold_data.min()) / (gold_data.max() - gold_data.min())
gas_data_normalized = (gas_data - gas_data.min()) / (gas_data.max() - gas_data.min())

# Convert the data into a stationary series by differencing
gold_data_stationary = gold_data_normalized.diff().dropna()
gas_data_stationary = gas_data_normalized.diff().dropna()

# Save the modified data to CSV files
gold_data_stationary.to_csv('D:\Sharif University of Tech\Term 4\AP\Project\Third faze\Data_Analysis\CSV Files\gold_data_stationary.csv')
gas_data_stationary.to_csv('D:\Sharif University of Tech\Term 4\AP\Project\Third faze\Data_Analysis\CSV Files\gas_data_stationary.csv')

# Trend Analysis
window_size = 12
gold_data_ma = gold_data['Close'].rolling(window=window_size).mean()
gas_data_ma = gas_data['Close'].rolling(window=window_size).mean()

# Seasonality Analysis
gold_data_seasonal = gold_data['Close'] - gold_data_ma
gas_data_seasonal = gas_data['Close'] - gas_data_ma

# Visualization: Plot each time series
plt.figure(figsize=(10, 6))
plt.plot(gold_data.index, gold_data['Close'], label='Gold')
plt.plot(gold_data.index, gold_data_ma, label='Gold Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gold Price Over Time')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(gas_data.index, gas_data['Close'], label='Gas')
plt.plot(gas_data.index, gas_data_ma, label='Gas Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Gas Price Over Time')
plt.legend()
plt.show()

# Descriptive Statistics: Calculate key statistics
gold_stats = gold_data.describe()
gas_stats = gas_data.describe()

print("Gold Data Statistics:")
print(gold_stats)

print("\nGas Data Statistics:")
print(gas_stats)

# Causality and Correlation Analysis: Investigate relationships between different variables using cross-correlation analysis
correlation = gold_data['Close'].corr(gas_data['Close'])
print("\nCorrelation between Gold and Gas prices:", correlation)

# Trend Analysis Results
print("\nTrend Analysis - Gold:")
print(gold_data_ma.tail())

print("\nTrend Analysis - Gas:")
print(gas_data_ma.tail())

# Seasonality Analysis Results
print("\nSeasonality Analysis - Gold:")
print(gold_data_seasonal.tail())

print("\nSeasonality Analysis - Gas:")
print(gas_data_seasonal.tail())
