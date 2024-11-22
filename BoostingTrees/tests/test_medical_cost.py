#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class DecisionTreeRegressor:
    """
    A simple decision tree regressor for fitting residuals.
    """
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def _split(self, X, y):
        """
        Find the best split for a dataset.
        """
        best_split = {"feature": None, "threshold": None, "loss": float("inf")}
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_residuals = y[left_mask]
                right_residuals = y[right_mask]
                
                # Mean squared error as loss
                loss = (
                    np.sum((left_residuals - np.mean(left_residuals)) ** 2) +
                    np.sum((right_residuals - np.mean(right_residuals)) ** 2)
                )
                
                if loss < best_split["loss"]:
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "loss": loss,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }
        
        return best_split

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        if depth >= self.max_depth or len(set(y)) == 1:
            return {"value": np.mean(y)}

        split = self._split(X, y)
        if split["feature"] is None:
            return {"value": np.mean(y)}

        left_tree = self._build_tree(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right_tree = self._build_tree(X[split["right_mask"]], y[split["right_mask"]], depth + 1)

        return {
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, x, tree):
        """
        Predict a single sample using the tree.
        """
        if "value" in tree:
            return tree["value"]
        
        feature = tree["feature"]
        threshold = tree["threshold"]

        if x[feature] <= threshold:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class GradientBoostingTree:
    """
    Gradient Boosting Tree implementation with explicit gamma calculation.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss="squared_error"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None
        self.loss = loss

    def _gradient(self, y, y_pred):
        """
        Compute the gradient of the loss function.
        """
        if self.loss == "squared_error":
            return y - y_pred
        raise ValueError("Unsupported loss function")

    def _gamma(self, residuals, region):
        """
        Compute the optimal gamma for a region as per Equation (10.30).
        """
        return np.mean(residuals[region])

    def fit(self, X, y):
        """
        Train the gradient boosting tree model.
        """
        self.init_prediction = np.mean(y)  # Start with the mean prediction
        predictions = np.full_like(y, self.init_prediction, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradients)
            residuals = self._gradient(y, predictions)

            # Train a decision tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions with the tree's contribution
            tree_predictions = tree.predict(X)

            for region in np.unique(tree_predictions):
                mask = tree_predictions == region
                gamma = self._gamma(residuals, mask)
                predictions[mask] += self.learning_rate * gamma

    def predict(self, X):
        """
        Predict target values for input data X.
        """
        predictions = np.full((X.shape[0],), self.init_prediction, dtype=np.float64)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions


# Example Usage
if __name__ == "__main__":
    # Import necessary libraries
    import numpy as np

    # Generate synthetic regression data
    def make_synthetic_regression(n_samples=100, n_features=7, noise=0.1, random_state=42):
        np.random.seed(random_state)
        X = np.random.rand(n_samples, n_features)  # Features: random values in [0, 1]
        coefficients = np.random.rand(n_features)  # Random coefficients for linear relation
        y = X @ coefficients + noise * np.random.randn(n_samples)  # Linear relationship + noise
        return X, y

    # Compute mean squared error manually
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Generate data
    X, y = make_synthetic_regression(n_samples=100, n_features=7, noise=0.1, random_state=42)
    y = y / np.std(y)  # Normalize target for simplicity

    # Train Gradient Boosting Tree
    model = GradientBoostingTree(n_estimators=50, learning_rate=0.1, max_depth=3, loss="squared_error")
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    # Evaluate
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse:.4f}")

    print("Predictions for new data:", predictions[:10])  # Display first 10 predictions


# In[2]:


#Dataset 3
import pandas as pd
import numpy as np

# Load Medical Cost dataset
file_path = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
data = pd.read_csv(file_path)

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Features and Target
X = data.drop('charges', axis=1).values  # Convert features to NumPy array
y = data['charges'].values  # Convert target to NumPy array

# Normalize the target variable (y)
y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std  # Normalize

# Manual Train-Test Split (80% Train, 20% Test)
n_samples = X.shape[0]
split_ratio = 0.8  # 80% train, 20% test
split_index = int(n_samples * split_ratio)

# Shuffle indices
indices = np.arange(n_samples)
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)

# Split the data
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Train Gradient Boosting Tree
model = GradientBoostingTree(n_estimators=50, learning_rate=0.1, max_depth=3, loss="squared_error")
model.fit(X_train, y_train)

# Predict on Test Data
predictions = model.predict(X_test)

# Evaluate
mse = np.mean((y_test - predictions) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.4f}")

# Rescale Predictions Back to Original Scale
predictions_original_scale = predictions * y_std + y_mean

# Print example predictions
print("Predictions (original scale):", predictions_original_scale[:10])


# In[ ]:




