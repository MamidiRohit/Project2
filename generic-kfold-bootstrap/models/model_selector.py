import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression


class ModelSelector:
    def __init__(self, model, k=5, n_bootstraps=100, random_state=None):
        """
        model: A model instance to be evaluated (e.g., LinearRegression()).
        k: Number of folds for k-fold cross-validation.
        n_bootstraps: Number of bootstrap samples.
        random_state: Seed for reproducibility.
        """
        self.model = model
        self.k = k
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

    def k_fold_cv(self, X, y):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        cv_errors = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            cv_errors.append(mean_squared_error(y_test, y_pred))

        return np.mean(cv_errors)

    def bootstrap(self, X, y):
        bootstrap_errors = []

        for _ in range(self.n_bootstraps):
            X_train, y_train = resample(X, y, random_state=self.random_state)
            X_test, y_test = resample(X, y, random_state=self.random_state)
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            bootstrap_errors.append(mean_squared_error(y_test, y_pred))

        return np.mean(bootstrap_errors)

    
    def calculate_aic(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        n = len(y)
        p = X.shape[1] + 1  # Number of features + intercept
        aic_score = n * np.log(mse) + 2 * p
        return aic_score

    def calculate_bic(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        n = len(y)
        p = X.shape[1] + 1  # Number of features + intercept
        bic_score = n * np.log(mse) + p * np.log(n)
        return bic_score

    def evaluate(self, X, y):
        cv_score = self.k_fold_cv(X, y)
        bootstrap_score = self.bootstrap(X, y)
        aic_score = self.calculate_aic(X, y)
        bic_score = self.calculate_bic(X, y)
        
        return {
            "k_fold_cv": cv_score,
            "bootstrap": bootstrap_score,
            "aic": aic_score,
            "bic": bic_score
        }
