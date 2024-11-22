import numpy as np
import pandas as pd
import csv
# Linear Regression Implementation
class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Train the model using the normal equation.
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
        # Adding a small value to the diagonal of X^T X to prevent singular matrix issues
        lambda_reg = 1e-8  # Regularization term
        XTX = X.T @ X + lambda_reg * np.eye(X.shape[1])  # Regularized X^T X
        self.weights = np.linalg.inv(XTX) @ X.T @ y

    def predict(self, X):
        """
        Predict using the fitted model.
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
        return X @ self.weights


# Metrics Implementation
def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Compute R-squared (R²), with a small epsilon to avoid division by zero.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    epsilon = 1e-10  # Small value to prevent division by zero
    return 1 - (ss_residual / (ss_total + epsilon))



# k-Fold Cross-Validation
def k_fold_cross_validation(model, X, y, k, shuffle):
    """
    Perform k-fold cross-validation and calculate MSE, MAE, and R².
    """
    metrics = {"mse": [], "mae": [], "r2": []}
    n = len(X)
    fold_size = n // k

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics["mse"].append(mean_squared_error(y_val, y_pred))
        metrics["mae"].append(mean_absolute_error(y_val, y_pred))
        metrics["r2"].append(r2_score(y_val, y_pred))

    averages = {key: np.mean(value) for key, value in metrics.items()}
    return metrics, averages


# Bootstrapping Validation
def bootstrapping(model, X, y, s, epochs):
    """
    Perform bootstrapping validation and calculate MSE, MAE, and R².
    """
    metrics = {"mse": [], "mae": [], "r2": []}
    n = len(X)

    for _ in range(epochs):
        indices = np.random.choice(range(n), size=s, replace=True)
        X_train, y_train = X[indices], y[indices]
        out_of_sample = [i for i in range(n) if i not in indices]
        X_val, y_val = X[out_of_sample], y[out_of_sample]

        model.fit(X_train, y_train)
        if len(out_of_sample) > 0:
            y_pred = model.predict(X_val)

            metrics["mse"].append(mean_squared_error(y_val, y_pred))
            metrics["mae"].append(mean_absolute_error(y_val, y_pred))
            metrics["r2"].append(r2_score(y_val, y_pred))

    averages = {key: np.mean(value) for key, value in metrics.items()}
    return metrics, averages


def generate_data(n_samples, n_features):
    """
    Generate synthetic data for testing.
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    true_weights = np.random.rand(n_features + 1)
    y = X @ true_weights[1:] + true_weights[0] + np.random.normal(0, 0.1, n_samples)
    return X, y
