import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

st.title("📈 Stock Trend Predictor")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

# Load data
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Feature engineering
data["SMA_10"] = data["Close"].rolling(10).mean()
data["SMA_50"] = data["Close"].rolling(50).mean()
data["Returns"] = data["Close"].pct_change()
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

data = data.dropna()

# Model
features = ["SMA_10", "SMA_50", "Returns"]
X = data[features]
y = data["Target"]

model = RandomForestClassifier()
model.fit(X, y)

# Prediction
latest = X.iloc[-1:].values
prediction = model.predict(latest)[0]

# Output
st.subheader("Prediction for Next Day:")
if prediction == 1:
    st.success("📈 Stock will go UP")
else:
    st.error("📉 Stock will go DOWN")

# Plot
st.subheader("Stock Price Chart")

fig, ax = plt.subplots()
ax.plot(data["Close"], label="Close Price")
ax.plot(data["SMA_10"], label="SMA 10")
ax.plot(data["SMA_50"], label="SMA 50")
ax.legend()

st.pyplot(fig)