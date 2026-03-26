from src.data_loader import load_data
from src.features import create_features
from src.model import train_model, evaluate

from sklearn.model_selection import train_test_split

def main():
    data = load_data()
    data = create_features(data)

    features = ["SMA_10", "SMA_50", "Returns"]
    X = data[features]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = train_model(X_train, y_train)
    acc = evaluate(model, X_test, y_test)

    print(f"Model Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()