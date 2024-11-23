import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add a bias column to the input data
        X = np.c_[np.ones(X.shape[0]), X]  
        # Closed-form solution for linear regression
        XTX = X.T @ X
        XTy = X.T @ y
        self.weights = np.linalg.solve(XTX, XTy)

    def predict(self, X):
        # Add bias term for prediction
        X = np.c_[np.ones(X.shape[0]), X]  
        return X @ self.weights


class RegressionMetrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def r_squared(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


    @staticmethod
    def aic(y_true, y_pred, n_features):
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        k = n_features + 1  # Adding bias as a parameter
        return n * np.log(rss / n) + 2 * k
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

