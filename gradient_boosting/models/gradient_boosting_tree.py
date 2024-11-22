import numpy as np
from models.decision_tree_stump import DecisionTreeStump

class GradientBoostingTree:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10):
        """
        Initialize the Gradient Boosting Tree.

        Parameters:
        - n_estimators: Number of boosting stages (default is 100).
        - learning_rate: Shrinkage parameter to scale the contribution of each tree (default is 0.1).
        - max_depth: Maximum depth of individual decision tree stumps (default is 3).
        - min_samples_split: Minimum number of samples required to split an internal node (default is 10).
        """
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.max_depth = max_depth  
        self.min_samples_split = min_samples_split  
        self.models = [] 
        self.init_prediction = None  

    def fit(self, X, y):
        """
        Train the Gradient Boosting Tree model.

        Parameters:
        - X: Feature matrix (n_samples x n_features).
        - y: Target values (n_samples,).
        """
        # Initialize predictions with the mean of the target values
        self.init_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.init_prediction, dtype=np.float64) 

        for _ in range(self.n_estimators):  # Iterate over the number of boosting stages
            # Compute residuals (negative gradient of loss function)
            residuals = y - y_pred

            # Train a decision tree stump to fit the residuals
            tree = DecisionTreeStump(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)

            # Predict residuals using the current tree and update overall predictions
            predictions = tree.predict(X)
            y_pred += self.learning_rate * predictions  # Update predictions with scaled contribution from the tree

            # Add the trained tree to the model list
            self.models.append(tree)

    def predict(self, X):
        """
        Predict target values for the input data.

        Parameters:
        - X: Feature matrix (n_samples x n_features).

        Returns:
        - Predicted target values (n_samples,).
        """
        # Start predictions with the initial mean value
        y_pred = np.full((X.shape[0],), self.init_prediction, dtype=np.float64)

        # Aggregate predictions from each fitted tree
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X) 
        
        return y_pred
