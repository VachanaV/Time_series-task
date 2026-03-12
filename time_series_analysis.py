import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error

# 1 Load dataset
df = pd.read_csv("airline-passengers.csv", parse_dates=["Month"], index_col="Month")

print("Dataset Preview")
print(df.head())

# 2 Plot raw data
plt.figure(figsize=(10,5))
plt.plot(df)
plt.title("Monthly Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()

# 3 Time Series Decomposition
decomposition = seasonal_decompose(df, model='multiplicative')

fig = decomposition.plot()
fig.set_size_inches(10,8)
plt.show()

# 4 Moving Averages
df["MA6"] = df["Passengers"].rolling(window=6).mean()
df["MA12"] = df["Passengers"].rolling(window=12).mean()

plt.figure(figsize=(10,5))
plt.plot(df["Passengers"], label="Actual")
plt.plot(df["MA6"], label="6 Month MA")
plt.plot(df["MA12"], label="12 Month MA")

plt.legend()
plt.title("Moving Average Smoothing")
plt.show()

# 5 Split dataset
train = df.iloc[:-12]
test = df.iloc[-12:]

# 6 ARIMA Model
model = ARIMA(train["Passengers"], order=(2,1,1))
model_fit = model.fit()

# 7 Forecast
forecast = model_fit.forecast(steps=12)

# 8 Model Evaluation
rmse = np.sqrt(mean_squared_error(test["Passengers"], forecast))
print("RMSE:", rmse)

# 9 Plot Forecast
plt.figure(figsize=(10,5))

plt.plot(train.index, train["Passengers"], label="Training Data")
plt.plot(test.index, test["Passengers"], label="Actual")
plt.plot(test.index, forecast, label="Forecast")

plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()