import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example list of daily prices
daily_prices = [1900.23, 1900.6, 1897.34, 1883.65, 1855.19, 1859.71, 1874.84, 1864.4, 1868.18, 1928.81, 1991.8, 1973.18,
                2037.2, 2047.46, 2066.01, 2069.34, 2134.87, 2148.93, 2166.25, 2215.34, 2203.84, 2152.04, 2185.72, 2175.29,
                2184.77, 2166.05, 2175.67, 2186.5, 2185.45, 2166.31, 2130.29, 2111.43, 2118.55, 2096.69, 2096.32, 2101.24,
                2181.61, 2171.17, 2180.2, 2122.41, 2105.03, 2090.48, 2069.66, 2049.65, 2078.47, 2016.53, 2029.23, 2024.81,
                2020.71, 1995.33, 1967.49, 1937.42, 1955.56, 2008.26, 1975.43, 1950.41, 1953.95, 1938.62, 1938.62, 1933.23,
                1926.27, 1952.18, 1933.11, 1915.32, 1885.65, 1910.34, 1920.09, 1920.01, 1910.49, 1910.49, 1888.61, 1872.65,
                1866.82, 1875.87, 1867.46, 1837.65, 1845.64, 1871.46, 1973.08, 1973.08, 1934.5, 1931.83, 1905.96, 1903.35,
                1918.48, 1904.94, 1885.77, 1887.29, 1907.54, 1892.94, 1884.94, 1871.63, 1866.68, 1856.96, 1870.4, 1850.72,
                1831.63, 1845.64, 1840.05, 1840.82, 1821.8, 1817.86, 1821.53, 1846.69, 1878.68, 1883.44, 1891.36, 1943.8,
                1954.3, 1944.03, 1884.48, 1850.79, 1866.34, 1922.13, 1902.75, 1929.16, 1933.39, 1941.52, 1984.36, 1989.42,
                2020.15, 2040.57, 2045.05, 2097.53, 2093.55, 2067.21, 2054.62, 2057.37, 2124.0, 2138.47, 2193.51, 2154.8,
                2096.72, 2079.42, 2090.37, 2050.28, 2050.35, 2081.75, 2078.24, 2072.24, 2095.72, 2085.68, 2097.98, 2147.68,
                2154.02, 2165.7]

# Creating a DataFrame
df = pd.DataFrame({'Date': pd.date_range('2022-01-01', periods=len(daily_prices)),
                   'Price': daily_prices})

# Setting the 'Date' column as the index
df.set_index('Date', inplace=True)

# Visualize historical data
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Price'], label='Historical Prices', color='orange')

# Data preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

# Create sequences for LSTM
seq_length = 10
dataX, dataY = [], []
for i in range(len(scaled_data) - seq_length):
    a = scaled_data[i:(i + seq_length), 0]
    dataX.append(a)
    dataY.append(scaled_data[i + seq_length, 0])
dataX, dataY = np.array(dataX), np.array(dataY)

# Reshape input to be [samples, time steps, features]
dataX = np.reshape(dataX, (dataX.shape[0], 1, dataX.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, seq_length), return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(dataX, dataY, epochs=100, batch_size=1, verbose=2)

# Predict next week's prices
future_data = scaled_data[-seq_length:].reshape((1, 1, seq_length))
predicted_data = []

for i in range(7):
    prediction = model.predict(future_data)
    predicted_data.append(prediction[0, 0])
    future_data = np.roll(future_data, -1, axis=2)
    future_data[0, 0, -1] = prediction[0, 0]

# Invert the scaler to get the absolute price data
predicted_data = scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))

# Plot predicted data
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
plt.plot(future_dates, predicted_data, label='Predicted Prices', color='purple')

# Finalize plot
plt.title('Historical and Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
