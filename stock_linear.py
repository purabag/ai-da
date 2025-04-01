import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timezone  # Import timezone from datetime

# Get stock symbol as user input
stock_symbol = input("Enter the stock symbol (e.g., 'PLTR'): ").strip().upper()

# Fetch real-time stock price data
today_date = pd.Timestamp.today().strftime('%Y-%m-%d')  # Get today's date dynamically
df = yf.download(stock_symbol, start="2020-01-01", end=today_date)  # Fetch latest data

# Check if the data is empty
if df.empty:
    print(f"Error: Could not fetch data for {stock_symbol}. Please try again later.")
    exit()

# Prepare Data
df = df[['Close']].reset_index()  # Use only closing prices
df['Date'] = pd.to_datetime(df['Date'])  # Convert Date back to datetime format

# Log-transform the closing prices for better handling of volatility
df['Log_Close'] = np.log(df['Close'])

# Convert dates to UNIX timestamps for regression
df['Timestamp'] = df['Date'].astype('int64') // 10**9  # Convert to UNIX timestamp

X = df[['Timestamp']].values  # Independent variable (Date)
y = df['Log_Close'].values  # Ensure y is 1D

# Check if the dataset has enough data for splitting
if X.shape[0] == 0:
    print("No data available for this stock symbol.")
    exit()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)  # Fit model

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")

# Predict the next day's closing price
latest_timestamp = df['Timestamp'].max()  # Get latest date's timestamp
next_day_timestamp = latest_timestamp + 86400  # Add one day (86400 seconds)

predicted_log_next_day_price = model.predict([[next_day_timestamp]])[0]  # Predict log-transformed price
predicted_next_day_price = np.exp(predicted_log_next_day_price)  # Convert back from log scale

# Convert timestamp to UTC
predicted_date = datetime.fromtimestamp(next_day_timestamp, timezone.utc).strftime('%Y-%m-%d')

print(f"Predicted closing price for {stock_symbol} on {predicted_date} (UTC): ${predicted_next_day_price:.2f}")

# Convert timestamps back to dates for plotting
X_test_dates = pd.to_datetime(X_test.flatten(), unit='s')  # Convert test dates back to datetime
X_train_dates = pd.to_datetime(X_train.flatten(), unit='s')  # Convert train dates back to datetime

# Visualize the Predictions
plt.figure(figsize=(10, 5))
plt.scatter(X_train_dates, np.exp(y_train), color='green', label='Actual Prices(Training)', alpha=0.5)
plt.scatter(X_test_dates, np.exp(y_test), color='blue', label='Actual Prices(Testing)')
plt.plot(X_test_dates, np.exp(y_pred), color='red', linewidth=2, label='Predicted Prices')

plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.title(f"Stock Price Prediction for {stock_symbol}")
plt.xticks(rotation=0)  # Rotate for better readability
plt.legend()
plt.show()