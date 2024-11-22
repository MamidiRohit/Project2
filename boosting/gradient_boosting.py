import numpy as np

class GradientBoostingTree:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_pred = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.initial_pred = np.mean(y)
        y_pred = np.full(n_samples, self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.mean(y)
        
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)

        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float("inf")
        best_split = None
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_mse = self._compute_mse(y[left_mask])
                right_mse = self._compute_mse(y[right_mask])
                mse = (left_mask.sum() * left_mse + right_mask.sum() * right_mse) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature_idx, threshold)
        return best_split

    def _compute_mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        feature_idx = tree["feature_idx"]
        threshold = tree["threshold"]
        
        if x[feature_idx] <= threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])
