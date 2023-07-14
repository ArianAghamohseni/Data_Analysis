import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



# Frist part :


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


# PCA handeling :


# Downloading and preprocessing data
exchange_rate_data = yf.download(['EURUSD=X', 'SAR=X', 'CNY=X'], start='2010-01-01', end='2020-12-31')
bitcoin_data = yf.download('BTC-USD', start='2010-01-01', end='2020-12-31')
interest_rates_data = yf.download('^IRX', start='2010-01-01', end='2020-12-31')
gold_data = yf.download('GLD', start='2010-01-01', end='2020-12-31')
silver_data = yf.download('SLV', start='2010-01-01', end='2020-12-31')
wheat_data = yf.download('ZW=F', start='2010-01-01', end='2020-12-31')

# Data sampling and alignment
exchange_rate_data = exchange_rate_data.resample('M').last()
bitcoin_data = bitcoin_data.resample('M').last()
interest_rates_data = interest_rates_data.resample('M').last()
gold_data = gold_data.resample('M').last()
silver_data = silver_data.resample('M').last()
wheat_data = wheat_data.resample('M').last()

# Handling null data
exchange_rate_data = exchange_rate_data.fillna(method='ffill')
bitcoin_data = bitcoin_data.fillna(method='ffill')
interest_rates_data = interest_rates_data.fillna(method='ffill')
gold_data = gold_data.fillna(method='ffill')
silver_data = silver_data.fillna(method='ffill')
wheat_data = wheat_data.fillna(method='ffill')

# Concatenate the data
concatenated_data = pd.concat([exchange_rate_data, bitcoin_data, interest_rates_data,
                               gold_data, silver_data, wheat_data], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(concatenated_data)

# Data normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(imputed_data)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(normalized_data)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Biplot
plt.figure(figsize=(10, 6))
for i in range(len(pca.components_)):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
    plt.text(pca.components_[0, i] * 1.2, pca.components_[1, i] * 1.2, concatenated_data.columns[i], color='g', ha='center', va='center')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Biplot')
plt.grid()
plt.show()





# Second part :


