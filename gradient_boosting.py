import numpy as np

# Define the function to calculate the gradient of the squared loss
def squared_loss_gradient(y, f):
    """
    Compute the gradient for the squared loss function.

    Parameters:
    - y (np.array): The target values.
    - f (np.array): The predicted values.

    Returns:
    - np.array: The gradient of the squared loss.
    """
    return y - f

# Define the Node class to represent each node in the decision tree
class Node:
    """
    A node in the decision tree.

    Attributes:
    - value (float): The value at the node, used for leaf nodes.
    - left (Node): Left child node.
    - right (Node): Right child node.
    - threshold (float): The threshold for splitting.
    - feature (int): The index of the feature used for splitting.
    """
    def __init__(self, value=None, left=None, right=None, threshold=None, feature=None):
        self.value = value
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature

# Define the DecisionTree class for building the regression tree
class DecisionTree:
    """
    A simple decision tree for regression.
    
    Attributes:
    - max_depth (int): The maximum depth of the tree.
    - root (Node): The root node of the tree.
    """
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, residuals):
        """
        Fit the decision tree to the residuals.

        Parameters:
        - X (np.array): Feature matrix.
        - residuals (np.array): Residuals to fit.
        """
        self.root = self._build_tree(X, residuals, depth=0)

    def _build_tree(self, X, residuals, depth):
        """
        Recursively build the decision tree.
        
        Parameters:
        - X (np.array): Feature matrix.
        - residuals (np.array): Residuals to fit.
        - depth (int): Current depth of the tree.
        
        Returns:
        - Node: The constructed tree node.
        """
        num_samples = X.shape[0]
        if depth >= self.max_depth or num_samples <= 1:
            leaf_value = np.mean(residuals)
            return Node(value=leaf_value)
        
        best_feature, best_threshold, best_var = None, None, np.inf
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_var = np.var(residuals[left_mask])
                right_var = np.var(residuals[right_mask])
                total_var = left_var + right_var
                if total_var < best_var:
                    best_feature, best_threshold, best_var = feature, threshold, total_var

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_node = self._build_tree(X[left_mask], residuals[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], residuals[right_mask], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def predict(self, X):
        """
        Make predictions using the decision tree.

        Parameters:
        - X (np.array): Feature matrix.

        Returns:
        - np.array: Predicted values.
        """
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        """
        Recursively predict by traversing the decision tree.

        Parameters:
        - x (np.array): Single feature vector.
        - node (Node): Current node of the tree.

        Returns:
        - float: Predicted value.
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

# Define the GradientBoosting class for boosting decision trees
class GradientBoosting:
    """
    Gradient Boosting for regression.

    Attributes:
    - n_estimators (int): Number of boosting stages to perform.
    - learning_rate (float): Learning rate shrinks the contribution of each tree.
    - max_depth (int): Maximum depth of each decision tree.
    - models (list): List of successive decision tree models.
    - initial_prediction (float): Initial prediction to start the boosting.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model.

        Parameters:
        - X (np.array): Feature matrix.
        - y (np.array): Target values.
        """
        # Initialize the first model to the mean of y
        self.initial_prediction = np.mean(y)
        f_m = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - f_m
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            f_m += self.learning_rate * predictions
            self.trees.append(tree)  # Store the tree instead of predictions

    def predict(self, X):
        """
        Make predictions using the boosted model.

        Parameters:
        - X (np.array): Feature matrix.

        Returns:
        - np.array: Predicted values.
        """
        # Start with the initial mean prediction
        f_m = np.full(X.shape[0], self.initial_prediction)

        # Accumulate predictions from each tree
        for tree in self.trees:
            f_m += self.learning_rate * tree.predict(X)

        return f_m
