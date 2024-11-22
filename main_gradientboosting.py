import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from boosting.gradient_boosting import GradientBoostingTree

def load_dataset(file_path):
    """
    Load a dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Dataset is empty. Please provide a valid dataset.")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

def preprocess_data(X, y):
    """
    Preprocess the dataset: scale features and return scaled X and y.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std

    return X_scaled, y_scaled

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using MSE, MAE, and R² metrics.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, mae, r2

def plot_results(y_true, y_pred):
    """
    Plot actual vs. predicted values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Load Dataset
    dataset_path = "data/diabetes_dataset.csv"
    X, y = load_dataset(dataset_path)

    # 2. Preprocess Data
    X, y = preprocess_data(X, y)

    # 3. Split Data into Training and Testing Sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. Train Gradient Boosting Model
    print("Training Gradient Boosting Model...")
    model = GradientBoostingTree(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    print("Model training complete!")

    # 5. Make Predictions
    y_pred = model.predict(X_test)

    # 6. Evaluate the Model
    mse, mae, r2 = evaluate_model(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 7. Plot Results
    print("\nPlotting Actual vs. Predicted Results...")
    plot_results(y_test, y_pred)
