import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import from Model
from Model.gradient_boosting import GradientBoostingTree
from Model.ModelSelection import grid_search_max_depth,calculate_aic


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
    Preprocess the dataset: scale features and return scaled X and y without using sklearn.
    """
    # Calculate mean and standard deviation for each feature
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    
    # Scale features
    X_scaled = (X - X_mean) / X_std

    # Optional: Scale target variable if needed
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std

    return X_scaled, y_scaled

def evaluate_model(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    return mse, mae, r2

def calculate_f1(y_true, y_pred, threshold=0.5):
    """
    Calculate F1 score by converting regression outputs to binary predictions.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = (y_true >= threshold).astype(int)

    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1, precision, recall


def plot_results(y_true, y_pred):
    """
    Plot Actual vs. Predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title("Actual vs. Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir,'highly_correlated_dataset.csv')
    X, y = load_dataset(dataset_path)
    X, y = preprocess_data(X, y)

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    #Train the parameter
    gbt_model = GradientBoostingTree(n_estimators=100, learning_rate=0.1, max_depth=2)
    gbt_model.fit(X_train, y_train)
    y_pred = gbt_model.predict(X_test)



    # Evaluate Model for Gradient Boosting
    # mse_samples, mse_mean = gbt_model.bootstrap(B=100)
    aic_value = calculate_aic(X_train,y_train)
    mse, mae, r2 = evaluate_model(y_test, y_pred)
    f1, precision, recall = calculate_f1(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AIC-Score: {aic_value:.4f}")


    # Plot Results
    plot_results(y_test, y_pred)
