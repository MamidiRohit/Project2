import numpy as np

class GradientBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Gradient Boosting Regressor from first principles.
        Parameters:
            - n_estimators: Number of boosting rounds.
            - learning_rate: Shrinkage parameter for updates.
            - max_depth: Maximum depth of individual trees.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_prediction = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        Parameters:
            - X: Features matrix of shape (n_samples, n_features).
            - y: Target vector of shape (n_samples,).
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty. Ensure X and y have valid values.")

        self.init_prediction = np.mean(y)
        residual = y - self.init_prediction

        for _ in range(self.n_estimators):
            tree = self._build_tree(X, residual, depth=0)
            self.models.append(tree)
            predictions = self._predict_tree(tree, X)
            residual -= self.learning_rate * predictions

    def predict(self, X):
        """
        Predict using the gradient boosting model.
        Parameters:
            - X: Features matrix of shape (n_samples, n_features).
        Returns:
            - Predictions for each sample.
        """
        if len(self.models) == 0:
            raise ValueError("The model has not been trained yet.")

        predictions = np.full(X.shape[0], self.init_prediction)
        for tree in self.models:
            predictions += self.learning_rate * self._predict_tree(tree, X)
        return predictions

    def _build_tree(self, X, residual, depth):
        """
        Recursively build a regression tree for fitting residuals.
        """
        if depth >= self.max_depth or len(residual) <= 1:
            return np.mean(residual) if len(residual) > 0 else 0  # Avoid empty slice issues

        feature, threshold, score = None, None, float("inf")
        for col in range(X.shape[1]):
            unique_values = np.unique(X[:, col])

            # Skip columns with all identical values (no split possible)
            if len(unique_values) == 1:
                continue

            for split in unique_values:
                left_mask = X[:, col] <= split
                right_mask = ~left_mask

                # Skip splits that produce empty nodes
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                left_mean = np.mean(residual[left_mask]) if np.any(left_mask) else 0
                right_mean = np.mean(residual[right_mask]) if np.any(right_mask) else 0

                error = (np.sum((residual[left_mask] - left_mean) ** 2) +
                         np.sum((residual[right_mask] - right_mean) ** 2))

                if error < score:
                    feature, threshold, score = col, split, error

        # Handle cases where no valid split is found
        if feature is None:
            return np.mean(residual)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], residual[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], residual[right_mask], depth + 1)

        return {"feature": feature, "threshold": threshold, "left": left_tree, "right": right_tree}

    def _predict_tree(self, tree, X):
        """
        Predict using a single decision tree.
        """
        if not isinstance(tree, dict):
            return np.full(X.shape[0], tree)

        feature = tree["feature"]
        threshold = tree["threshold"]
        left_mask = X[:, feature] <= threshold
        predictions = np.zeros(X.shape[0])
        predictions[left_mask] = self._predict_tree(tree["left"], X[left_mask]) if np.any(left_mask) else 0
        predictions[~left_mask] = self._predict_tree(tree["right"], X[~left_mask]) if np.any(~left_mask) else 0
        return predictions
