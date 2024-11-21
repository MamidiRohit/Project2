import pandas as pd
import numpy as np

class LinearRegressionModel:
    def fit(self, X, y):

        X = np.c_[np.ones(X.shape[0]), X] 
        X_transpose = X.T
        XtX = X_transpose @ X
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_transpose @ y
        self.weights = XtX_inv @ Xty

    def predict(self, X):

        X = np.c_[np.ones(X.shape[0]), X] 
        return X @ self.weights


class Metrics:    
    @staticmethod
    def mean_squared_error(y_true, y_pred):

        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def r_squared(y_true, y_pred):

        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_left = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_left / ss_total)
    
    @staticmethod
    def aic(y_true, y_pred, model, X):
        
        n = len(y_true)
        
        leftover_data = y_true - y_pred
        rss = np.sum(leftover_data**2)
        
        k = X.shape[1] + 1  
        
        aic = n * np.log(rss / n) + 2 * k
        
        return aic