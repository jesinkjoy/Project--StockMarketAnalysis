import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Title and Introduction
st.title("Stock Market Analysis Tool")

# Sidebar for user input
st.sidebar.header("User Input")

# Select stock ticker
ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL, MSFT, GOOGL)", "AAPL")

# Select date range
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

# Fetch stock data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

# Display data
st.subheader("Raw data")
st.write(data.tail())

# Statistical Summary
st.subheader("Statistical Summary")
st.write(data.describe())

# Plot closing price
st.subheader("Closing Price")
fig, ax = plt.subplots()
ax.plot(data["Date"], data["Adj Close"], label="Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Volume Analysis
st.subheader("Volume Analysis")
fig, ax = plt.subplots()
ax.plot(data["Date"], data["Volume"], label="Volume", color='orange')
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Moving Averages
st.subheader("Moving Averages")
ma_days = st.slider("Select number of days for moving average", 1, 100, 20)
data["MA"] = data["Adj Close"].rolling(window=ma_days).mean()
fig, ax = plt.subplots()
ax.plot(data["Date"], data["Adj Close"], label="Closing Price")
ax.plot(data["Date"], data["MA"], label=f"{ma_days}-Day MA")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Bollinger Bands
st.subheader("Bollinger Bands")
window = st.slider("Select window for Bollinger Bands", 1, 100, 20)
data['MA20'] = data['Adj Close'].rolling(window=window).mean()
data['stddev'] = data['Adj Close'].rolling(window=window).std()
data['Upper'] = data['MA20'] + (data['stddev'] * 2)
data['Lower'] = data['MA20'] - (data['stddev'] * 2)
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Adj Close'], label='Closing Price')
ax.plot(data['Date'], data['Upper'], label='Upper Band')
ax.plot(data['Date'], data['Lower'], label='Lower Band')
ax.fill_between(data['Date'], data['Lower'], data['Upper'], color='grey', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Time Series Decomposition
st.subheader("Time Series Decomposition")
data.set_index("Date", inplace=True)  # Set the date as index for decomposition
decomp = seasonal_decompose(data["Adj Close"], model="additive", period=365)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
decomp.observed.plot(ax=ax1, title="Observed")
decomp.trend.plot(ax=ax2, title="Trend")
decomp.seasonal.plot(ax=ax3, title="Seasonal")
decomp.resid.plot(ax=ax4, title="Residual")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ACF and PACF
st.subheader("Autocorrelation and Partial Autocorrelation")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['Adj Close'], ax=ax[0])
plot_pacf(data['Adj Close'], ax=ax[1])
plt.tight_layout()
st.pyplot(fig)

# Forecasting with Exponential Smoothing
st.subheader("Forecasting with Exponential Smoothing")
data.reset_index(inplace=True)  # Reset index before plotting
model = ExponentialSmoothing(data["Adj Close"], trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()
forecast = fit.forecast(steps=30)
forecast_index = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data["Date"], data["Adj Close"], label="Observed")
ax.plot(forecast_index, forecast, label="Forecast", linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Forecasting with ARIMA
st.subheader("Forecasting with ARIMA")
train_data = data["Adj Close"]

# Train ARIMA model
model_arima = ARIMA(train_data, order=(5, 1, 0))
fit_arima = model_arima.fit()

# Forecast future prices
forecast_arima = fit_arima.forecast(steps=30)
future_dates_arima = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data["Date"], data["Adj Close"], label="Observed")
ax.plot(future_dates_arima, forecast_arima, label="Forecast", linestyle='--')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Forecasting with SARIMA
st.subheader("Forecasting with SARIMA")
train_data = data["Adj Close"]

# Train SARIMA model
model_sarima = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
fit_sarima = model_sarima.fit(disp=False)

# Forecast future prices
forecast_sarima = fit_sarima.get_forecast(steps=30)
future_dates_sarima = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30)
forecast_sarima_ci = forecast_sarima.conf_int()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data["Date"], data["Adj Close"], label="Observed")
ax.plot(future_dates_sarima, forecast_sarima.predicted_mean, label="Forecast", linestyle='--')
ax.fill_between(future_dates_sarima, forecast_sarima_ci.iloc[:, 0], forecast_sarima_ci.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Calculate RMSE
train_size = int(len(train_data) * 0.8)
train, test = train_data[0:train_size], train_data[train_size:]
model_sarima_test = SARIMAX(train, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
fit_sarima_test = model_sarima_test.fit(disp=False)
predictions_sarima = fit_sarima_test.get_forecast(steps=len(test))
rmse_sarima = np.sqrt(mean_squared_error(test, predictions_sarima.predicted_mean))
st.write(f"Root Mean Squared Error (SARIMA): {rmse_sarima:.2f}")

# Compare RMSE of ARIMA and SARIMA
model_arima_test = ARIMA(train, order=(5, 1, 0))
fit_arima_test = model_arima_test.fit()
predictions_arima = fit_arima_test.forecast(steps=len(test))
rmse_arima = np.sqrt(mean_squared_error(test, predictions_arima))
st.write(f"Root Mean Squared Error (ARIMA): {rmse_arima:.2f}")



