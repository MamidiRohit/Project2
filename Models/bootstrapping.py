import numpy as np
from Models.LinearRegressionModel import LinearRegression, RegressionMetrics

class BootstrapModelSelection:
    def __init__(self, n_iterations=100, seed=42):
        self.n_iterations = n_iterations
        self.seed = seed

    def evaluate(self, X, y):
        np.random.seed(self.seed)
        n_samples = len(y)
        mse_scores, r2_scores, aic_scores, mae_scores, rmse_scores = [], [], [], [], []

        for i in range(self.n_iterations):
            bootstrap_indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
            oob_indices = np.setdiff1d(range(n_samples), bootstrap_indices)

            if len(oob_indices) == 0:
                continue

            X_train, X_oob = X[bootstrap_indices], X[oob_indices]
            y_train, y_oob = y[bootstrap_indices], y[oob_indices]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_oob)

            mse_scores.append(RegressionMetrics.mean_squared_error(y_oob, y_pred))
            r2_scores.append(RegressionMetrics.r_squared(y_oob, y_pred))
            aic_scores.append(RegressionMetrics.aic(y_oob, y_pred, X.shape[1]))
            mae_scores.append(RegressionMetrics.mean_absolute_error(y_oob, y_pred))
            rmse_scores.append(RegressionMetrics.root_mean_squared_error(y_oob, y_pred))

            print(f"Iteration {i + 1}/{self.n_iterations} - MSE: {mse_scores[-1]:.4f}, "
                  f"R-Squared: {r2_scores[-1]:.4f}, AIC: {aic_scores[-1]:.4f}, "
                  f"MAE: {mae_scores[-1]:.4f}, RMSE: {rmse_scores[-1]:.4f}")

        return {
            "mean_mse": np.mean(mse_scores),
            "mean_r2": np.mean(r2_scores),
            "mean_aic": np.mean(aic_scores),
            "mean_mae": np.mean(mae_scores),
            "mean_rmse": np.mean(rmse_scores),
        }
