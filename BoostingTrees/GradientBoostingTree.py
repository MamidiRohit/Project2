import numpy as np
from sklearn.metrics import mean_squared_error


class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  # Stores individual decision trees
        self.loss_history = []  # Tracks loss over iterations

    def fit(self, X, y):
        """
        Fits the gradient boosting model.
        """
        #starting with a constant value (mean of y)
        self.initial_prediction = np.mean(y)
        current_prediction = np.full_like(y, self.initial_prediction, dtype=float)

        for estimator in range(self.n_estimators):
            # Calculate residual errors to know how far we are from the actual target
            residuals = y - current_prediction

            # Training a tree to fix these residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # updating predictions
            update = tree.predict(X)
            current_prediction += self.learning_rate * update

            # Tracking loss
            loss = mean_squared_error(y, current_prediction)
            self.loss_history.append(loss)

            print(f"Iteration {estimator + 1}/{self.n_estimators}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predicts using the gradient boosting model.
        """
        current_prediction = np.full(X.shape[0], self.initial_prediction, dtype=float)
        for tree in self.trees:
            update = tree.predict(X)
            current_prediction += self.learning_rate * update
        return current_prediction

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        #For Building the decision tree.
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        #Predictng the target values for input data
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _build_tree(self, X, y, depth):
        #Recursive method to for constructing the decision tree
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split or np.std(y) == 0:
            return np.mean(y)

        best_split = self._find_best_split(X, y, num_features)
        if not best_split:
            return np.mean(y)

        left_indices = X[:, best_split["feature"]] < best_split["threshold"]
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree
        }

    def _find_best_split(self, X, y, num_features):
        best_split = {}
        min_error = float("inf")

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_error = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
                right_error = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)
                error = (len(y[left_indices]) * left_error + len(y[right_indices]) * right_error) / len(y)

                if error < min_error:
                    min_error = error
                    best_split = {"feature": feature, "threshold": threshold}

        return best_split if min_error < float("inf") else None

    def _predict_row(self, row, tree):
        #Predicting the target value for a single data row
        if isinstance(tree, dict):
            feature = tree["feature"]
            threshold = tree["threshold"]
            if row[feature] < threshold:
                return self._predict_row(row, tree["left"])
            else:
                return self._predict_row(row, tree["right"])
        else:
            return tree