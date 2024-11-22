# linear_regression.py

import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Adding bias term (intercept) to the features
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Closed-form solution to linear regression (Normal Equation)
        # w = (X^T X)^-1 X^T y
        X_transpose = X.T
        self.weights = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    def predict(self, X):
        # Adding bias term to test data
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights
