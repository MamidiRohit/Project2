import numpy as np
from Models.LinearRegressionModel import LinearRegression, RegressionMetrics

class KFoldCrossValidation:
    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed

    def evaluate(self, X, y):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(y))
        fold_size = len(y) // self.n_splits
        mse_scores, r2_scores, aic_scores, mae_scores, rmse_scores = [], [], [], [], []

        for fold in range(self.n_splits):
            val_indices = indices[fold * fold_size: (fold + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse_scores.append(RegressionMetrics.mean_squared_error(y_val, y_pred))
            r2_scores.append(RegressionMetrics.r_squared(y_val, y_pred))
            aic_scores.append(RegressionMetrics.aic(y_val, y_pred, X.shape[1]))
            mae_scores.append(RegressionMetrics.mean_absolute_error(y_val, y_pred))
            rmse_scores.append(RegressionMetrics.root_mean_squared_error(y_val, y_pred))

        return {
            "mean_mse": np.mean(mse_scores),
            "mean_r2": np.mean(r2_scores),
            "mean_aic": np.mean(aic_scores),
            "mean_mae": np.mean(mae_scores),
            "mean_rmse": np.mean(rmse_scores),
        }
