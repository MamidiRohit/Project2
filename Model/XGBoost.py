import numpy as np

def train_test_split(X, y, test_size=0.25, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int((1 - test_size) * len(indices))
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.initial_prediction = None

    def predict_proba(self, X):
        y_pred = self.predict(X)
        return 1 / (1 + np.exp(-y_pred))

    def _compute_pseudo_residuals(self, y, y_pred):
        pred_proba = 1 / (1 + np.exp(-y_pred))
        grad = pred_proba - y
        hess = pred_proba * (1 - pred_proba)
        return grad, hess

    def _split_gain(self, grad_sum_left, hess_sum_left, grad_sum_right, hess_sum_right):
        gain = (grad_sum_left**2 / (hess_sum_left + self.reg_lambda) +
                grad_sum_right**2 / (hess_sum_right + self.reg_lambda))
        return gain

    def _fit_tree(self, X, grad, hess, depth=0):
        n_samples, n_features = X.shape
        if depth == self.max_depth or n_samples <= 1:
            leaf_value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return {"leaf_value": leaf_value}

        best_gain = -np.inf
        best_split = None
        best_left_idx, best_right_idx = None, None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                grad_sum_left = np.sum(grad[left_idx])
                hess_sum_left = np.sum(hess[left_idx])
                grad_sum_right = np.sum(grad[right_idx])
                hess_sum_right = np.sum(hess[right_idx])
                gain = self._split_gain(grad_sum_left, hess_sum_left, grad_sum_right, hess_sum_right)
                if gain > best_gain and gain > self.gamma:
                    best_gain = gain
                    best_split = (feature_idx, threshold)
                    best_left_idx, best_right_idx = left_idx, right_idx

        if best_split is None:
            leaf_value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
            return {"leaf_value": leaf_value}

        feature_idx, threshold = best_split
        left_subtree = self._fit_tree(X[best_left_idx], grad[best_left_idx], hess[best_left_idx], depth + 1)
        right_subtree = self._fit_tree(X[best_right_idx], grad[best_right_idx], hess[best_right_idx], depth + 1)

        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _predict_tree(self, tree, X):
        if "leaf_value" in tree:
            return np.full(X.shape[0], tree["leaf_value"])
        feature_idx = tree["feature_idx"]
        threshold = tree["threshold"]
        left_idx = X[:, feature_idx] <= threshold
        right_idx = ~left_idx
        predictions = np.zeros(X.shape[0])
        predictions[left_idx] = self._predict_tree(tree["left"], X[left_idx])
        predictions[right_idx] = self._predict_tree(tree["right"], X[right_idx])
        return predictions

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        for i in range(self.n_estimators):
            grad, hess = self._compute_pseudo_residuals(y, y_pred)
            tree = self._fit_tree(X, grad, hess)
            self.trees.append(tree)
            y_pred += self.learning_rate * self._predict_tree(tree, X)
            if i % 10 == 0:
                mse = self.mean_squared_error(y, y_pred)
                print(f"Iteration {i}: Training MSE = {mse}")

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(tree, X)
        return y_pred

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def r2_score(y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)

    @staticmethod
    def gini_index(y_true, y_pred):
        y_true_sorted = y_true[np.argsort(y_pred)]
        n = len(y_true)
        cumulative_y = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        gini = 1 - 2 * np.sum(cumulative_y - cumulative_y / n)
        return gini

    @staticmethod
    def gini_impurity(y_true, y_pred):
        p = np.mean(y_true)
        return 2 * p * (1 - p)

    @staticmethod
    def f1_score(y_true, y_pred, threshold=0.5):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return f1