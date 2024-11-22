# model.py
import numpy as np
import matplotlib.pyplot as plt

class BaseRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

class LinearRegression(BaseRegression):
    def fit(self, X, y):
        X_b = self._add_bias(X)
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        return self
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

class RidgeRegression(BaseRegression):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def fit(self, X, y):
        X_b = self._add_bias(X)
        n_features = X_b.shape[1]
        identity = np.eye(n_features)
        identity[0, 0] = 0  # Don't regularize intercept
        theta = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * identity).dot(X_b.T).dot(y)
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        return self
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

class ModelSelector:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test    
        
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def aic(self, y_true, y_pred, k):
        n = len(y_true)
        mse = self.mse(y_true, y_pred)
        return n * np.log(mse) + 2 * k
    
    def k_fold_cv(self, model, k=6):
        fold_size = len(self.X_train) // k
        mse_scores = []
        
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            
            X_val = self.X_train[start_idx:end_idx]
            y_val = self.y_train[start_idx:end_idx]
            X_train_fold = np.concatenate([self.X_train[:start_idx], self.X_train[end_idx:]])
            y_train_fold = np.concatenate([self.y_train[:start_idx], self.y_train[end_idx:]])
            
            model_copy = type(model)()
            model_copy.fit(X_train_fold, y_train_fold)
            y_pred = model_copy.predict(X_val)
            mse_scores.append(self.mse(y_val, y_pred))
        
        return np.mean(mse_scores)
    
    def bootstrap(self, model, n_iterations=100):
        n_samples = len(self.X_train)
        mse_scores = []
        
        for _ in range(n_iterations):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = self.X_train[indices]
            y_boot = self.y_train[indices]
            
            model_copy = type(model)()
            model_copy.fit(X_boot, y_boot)
            y_pred = model_copy.predict(self.X_test)
            mse_scores.append(self.mse(self.y_test, y_pred))
        
        return np.mean(mse_scores)
    
    def evaluate_model(self, model):
        # Train the model
        model.fit(self.X_train, self.y_train)
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        results = {
            #'train_mse': self.mse(self.y_train, y_pred_train),
            'test_mse': self.mse(self.y_test, y_pred_test),
            'kfold_mse': self.k_fold_cv(model),
            'bootstrap_mse': self.bootstrap(model),
            'aic': self.aic(self.y_test, y_pred_test, len(model.coefficients)),
        }
        return results