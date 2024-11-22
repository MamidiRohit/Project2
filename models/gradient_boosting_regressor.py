import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # Check if the node is a leaf.
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if max_depth <= 0:
            raise ValueError("max_depth must be a positive integer.")
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Fit the decision tree on the data.
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be NumPy arrays.")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array.")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] < self.min_samples_split:
            raise ValueError("Number of samples is less than min_samples_split.")
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Recursively build the decision tree.
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        #Find the best feature and threshold to split on.
        if len(feat_idxs) == 0:
            raise ValueError("feat_idxs cannot be empty.")
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Calculate the reduction in variance from a potential split.
        parent_variance = np.var(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        var_left, var_right = np.var(y[left_idxs]), np.var(y[right_idxs])
        child_variance = (n_left / n) * var_left + (n_right / n) * var_right

        return parent_variance - child_variance

    def _split(self, X_column, threshold):
        # Split the data into left and right groups based on the threshold.
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        # Return the mean value for regression tasks.
        return np.mean(y)

    def predict(self, X):
        # Predict the labels for a dataset.
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # Traverse the tree recursively to make a prediction for a single sample.
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
class GradientBoostingRegressor:
    # Gradient Boosting Regressor for regression tasks.
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if not (0 < learning_rate <= 1):
            raise ValueError("learning_rate must be in the range (0, 1].")
        if max_depth <= 0:
            raise ValueError("max_depth must be a positive integer.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_prediction = None

    def fit(self, X, y):
        # Fit the gradient boosting regressor on the training data.
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array.")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array.")
        if len(y) == 0:
            raise ValueError("The target array 'y' is empty.")
        if len(np.unique(y)) == 1:
            print("Warning: All target values are identical. The model will predict a constant value.")
        
        self.init_prediction = np.mean(y)
        current_predictions = np.full(y.shape, self.init_prediction)

        for i in range(self.n_estimators):
            try:
                residuals = y - current_predictions

                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                tree.fit(X, residuals)
                self.trees.append(tree)

                predictions = tree.predict(X)
                current_predictions += self.learning_rate * predictions
            except Exception as e:
                print(f"Error during training for tree {i + 1}: {e}. Continuing with the next tree.")

    def predict(self, X):
        # Predict target values for the input data.
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("X has no samples.")

        predictions = np.full((X.shape[0],), self.init_prediction)

        for i, tree in enumerate(self.trees):
            try:
                predictions += self.learning_rate * tree.predict(X)
            except Exception as e:
                print(f"Error during prediction with tree {i + 1}: {e}. Input shape: {X.shape}")

        return predictions