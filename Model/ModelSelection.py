import numpy as np
from gradient_boosting import GradientBoostingTree


def k_fold_cv(model, X, y, k=5, metric="mse"):
    """
    Perform k-fold cross-validation.
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    scores = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        val_idx = folds[i]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if metric == "mse":
            score = np.mean((y_val - y_pred) ** 2)
        elif metric == "mae":
            score = np.mean(np.abs(y_val - y_pred))
        else:
            raise ValueError("Unsupported metric. Use 'mse' or 'mae'.")

        scores.append(score)

    return np.mean(scores)


def grid_search_max_depth(X, y, max_depth_values, n_estimators=100, learning_rate=0.1, k=5):
    """
    Perform grid search to find the best max_depth with k-fold cross-validation.
    Args:
        X: Feature matrix.
        y: Target vector.
        max_depth_values: List of max_depth values to test.
        n_estimators: Fixed number of estimators.
        learning_rate: Fixed learning rate.
        k: Number of folds.
    Returns:
        Best max_depth and its corresponding score.
    """
    best_score = float("inf")
    best_max_depth = None

    for max_depth in max_depth_values:
        model = GradientBoostingTree(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )
        score = k_fold_cv(model, X, y, k=k)
        print(f"Max Depth: {max_depth} -> Score: {score}")
        if score < best_score:
            best_score = score
            best_max_depth = max_depth

    return best_max_depth, best_score

# def bootstrap(self, B=100):
#         """
#         Perform bootstrapping for model evaluation.
#         Args:
#             B (int): Number of bootstrap iterations.
#         Returns:
#             mse_samples (list): Mean squared errors for each bootstrap sample.
#             mse_mean (float): Mean of the mean squared errors across all bootstrap samples.
#         """
#         n = len(self.y)
#         mse_samples = []

#         for _ in range(B):
#             indices = np.random.choice(np.arange(n), size=n, replace=True)
#             X_sample, y_sample = self.X[indices], self.y[indices]

#             beta = np.linalg.pinv(X_sample.T @ X_sample) @ X_sample.T @ y_sample
#             y_pred = self.X @ beta
#             mse_samples.append(np.mean((self.y - y_pred) ** 2))

#         return mse_samples, np.mean(mse_samples)

# def calculate_aic(self):
#     """
#     Calculate the Akaike Information Criterion (AIC).
#     Returns:
#         aic (float): AIC score for the linear regression model.
#     """
#     n, p = self.X.shape
#     beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
#     y_pred = self.X @ beta
#     rss = np.sum((self.y - y_pred) ** 2)
#     aic = n * np.log(rss / n) + 2 * p
#     return aic
