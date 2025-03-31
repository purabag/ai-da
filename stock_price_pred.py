import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Download stock price data
stock_symbol = "PLTR"  # Change to any stock symbol you want
df = yf.download(stock_symbol, start="2020-01-01", end="2025-01-01")

# Step 2: Prepare Data
df = df[['Close']].reset_index()  # Use only closing prices
df['Date'] = df['Date'].astype('int64') // 10**9  # Convert dates to timestamp for regression

X = df[['Date']].values  # Independent variable (Date)
y = df[['Close']].values  # Dependent variable (Stock Price)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 7: Visualize the Predictions
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.xlabel("Date (Timestamp)")
plt.ylabel("Stock Price ($)")
plt.title(f"Stock Price Prediction for {stock_symbol}")
plt.legend()
plt.show()
