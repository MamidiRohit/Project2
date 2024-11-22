import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from models.gradient_reg import GradientBoostingRegressor

def load_netflix_data(file_path):
    """
    Load Netflix data from a CSV file and preprocess it.
    """
    data = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = ["Open", "High", "Low", "Close"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract features and target
    data = data.dropna()  # Drop missing values
    X = data[["Open", "High", "Low"]].values  # Features
    y = data["Close"].values  # Target

    return X, y

def test_netflix_regressor(file_path):
    """
    Test GradientBoostingRegressor on the Netflix dataset.
    """
    X, y = load_netflix_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, min_samples_split=2)

    # Train the model
    print("Training Gradient Boosting Regressor...")
    gbr.fit(X_train, y_train)
    print("Training complete.")

    # Make predictions
    print("Making predictions...")
    y_pred = gbr.predict(X_test)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Value: {r2:.4f}")

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "tests/netflix.csv"

    print("Testing Gradient Boosting Regressor on Netflix Data...")
    test_netflix_regressor(dataset_path)