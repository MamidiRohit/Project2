import numpy as np

class DecisionTreeStump:
    def __init__(self, max_depth=3, min_samples_split=10):
        """
        Initialize the DecisionTreeStump with optional constraints.

        Parameters:
        - max_depth: The maximum depth the stump can reach (default is 3).
        - min_samples_split: Minimum number of samples required to split a node (default is 10).
        """
        self.feature_index = None  # Index of the feature to split on
        self.threshold = None  # Threshold value for the split
        self.left_value = None  # Predicted value for the left split
        self.right_value = None  # Predicted value for the right split
        self.max_depth = max_depth  # Maximum allowable depth for the tree
        self.min_samples_split = min_samples_split  # Minimum samples to allow a split

    def fit(self, X, residuals, depth=1):
        """
        Fit the decision stump to the given data by finding the best split.

        Parameters:
        - X: Feature matrix (n_samples x n_features).
        - residuals: Target residuals to minimize.
        - depth: Current depth of the stump in the tree (default is 1).
        """
        n_samples, n_features = X.shape

        # Check the required criteria: if max depth is reached or too few samples
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            # Make the stump a leaf node with a constant prediction
            self.left_value = self.right_value = residuals.mean()
            return

        best_mse = float("inf") 
        found_split = False  

        # Iterate over each feature to find the best split
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])  
            for threshold in thresholds:
                # Divide the data into left and right splits
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                # Skip invalid splits where one side has no data
                if not np.any(left_indices) or not np.any(right_indices):
                    continue

                # Calculate means for each split
                left_mean = residuals[left_indices].mean()
                right_mean = residuals[right_indices].mean()

                # Compute MSE for the current split
                mse_left = ((residuals[left_indices] - left_mean) ** 2).sum()
                mse_right = ((residuals[right_indices] - right_mean) ** 2).sum()
                mse = mse_left + mse_right

                # Update the best split if current split is better
                if mse < best_mse:
                    found_split = True
                    best_mse = mse
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

        # If no valid split is found, make the stump a constant prediction leaf node
        if not found_split:
            self.left_value = self.right_value = residuals.mean()

    def predict(self, X):
        """
        Make predictions using the fitted decision stump.

        Parameters:
        - X: Feature matrix (n_samples x n_features).

        Returns:
        - Predictions for each sample in X.
        """
        if self.feature_index is None or self.threshold is None:
            # Return constant predictions if no valid split was found during fitting
            return np.full(X.shape[0], self.left_value)

        # Predict based on the threshold for the selected feature
        return np.where(
            X[:, self.feature_index] <= self.threshold,
            self.left_value,
            self.right_value
        )
