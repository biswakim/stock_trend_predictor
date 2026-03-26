def create_features(data):
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["Returns"] = data["Close"].pct_change()

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    return data.dropna()