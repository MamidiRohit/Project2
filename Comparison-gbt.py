import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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

    return X_scaled, y_scaled, y_mean, y_std

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using MSE, MAE, and R² metrics.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, mae, r2

def plot_comparison(y_true, custom_pred, sklearn_pred):
    """
    Plot comparison of predictions from both models against true values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, custom_pred, alpha=0.7, label="Custom Model Predictions", color="blue")
    plt.scatter(y_true, sklearn_pred, alpha=0.7, label="Sklearn Model Predictions", color="green")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Comparison of Custom vs Sklearn Model Predictions")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Load Dataset
    dataset_path = "data/boston_housing.csv"
    X, y = load_dataset(dataset_path)

    # 2. Preprocess Data
    X, y, y_mean, y_std = preprocess_data(X, y)

    # 3. Split Data into Training and Testing Sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. Train Custom Gradient Boosting Model
    print("Training Custom Gradient Boosting Model...")
    custom_model = GradientBoostingTree(n_estimators=100, learning_rate=0.1, max_depth=2)
    custom_model.fit(X_train, y_train)
    print("Custom Model training complete!")

    # 5. Train Sklearn Gradient Boosting Model
    print("Training Sklearn Gradient Boosting Model...")
    sklearn_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42)
    sklearn_model.fit(X_train, y_train)
    print("Sklearn Model training complete!")

    # 6. Make Predictions
    custom_pred = custom_model.predict(X_test) * y_std + y_mean
    sklearn_pred = sklearn_model.predict(X_test) * y_std + y_mean
    y_test_original = y_test * y_std + y_mean

    # 7. Evaluate Both Models
    print("\nCustom Model Evaluation:")
    custom_mse, custom_mae, custom_r2 = evaluate_model(y_test_original, custom_pred)
    print(f"Mean Squared Error (MSE): {custom_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {custom_mae:.4f}")
    print(f"R² Score: {custom_r2:.4f}")

    print("\nSklearn Model Evaluation:")
    sklearn_mse, sklearn_mae, sklearn_r2 = evaluate_model(y_test_original, sklearn_pred)
    print(f"Mean Squared Error (MSE): {sklearn_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {sklearn_mae:.4f}")
    print(f"R² Score: {sklearn_r2:.4f}")

    # 8. Plot Results
    print("\nPlotting Comparison of Results...")
    plot_comparison(y_test_original, custom_pred, sklearn_pred)