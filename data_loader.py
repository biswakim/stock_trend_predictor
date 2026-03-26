import yfinance as yf

def load_data(ticker="AAPL", start="2018-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end)
    return data