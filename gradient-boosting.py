import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingTree:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss='squared_error'):
        """
        Initializing the Gradient Boosting Tree model.
        
        Parameters:
        - n_estimators: Number of trees (iterations).
        - learning_rate: Step size for updating predictions.
        - max_depth: Maximum depth of individual regression trees.
        - loss: Loss function ('squared_error' or 'absolute_error').
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.trees = []
        self.initial_prediction = None

    def _initialize_loss(self, y):
        """Initializing the loss by computing the constant prediction."""
        if self.loss == 'squared_error':
            return np.mean(y)
        elif self.loss == 'absolute_error':
            return np.median(y)
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    def _compute_gradient(self, y, preds):
        """Computing the gradient of the loss."""
        if self.loss == 'squared_error':
            return y - preds
        elif self.loss == 'absolute_error':
            return np.sign(y - preds)
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    def fit(self, X, y):
        """
        Training the model using gradient boosting.
        
        Parameters:
        - X: Feature matrix (NumPy array of shape [n_samples, n_features]).
        - y: Target values (NumPy array of shape [n_samples]).
        """
        # Initializing the base prediction
        self.initial_prediction = self._initialize_loss(y)
        preds = np.full(y.shape, self.initial_prediction)
        
        # Iteratively building trees and update predictions
        for _ in range(self.n_estimators):
            # Computing residuals (negative gradients)
            residuals = self._compute_gradient(y, preds)
            
            # Fitting a regression tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Updating predictions
            preds += self.learning_rate * tree.predict(X)

    def predict(self, X):
        """
        Predicting the target values for a given feature matrix.
        
        Parameters:
        - X: Feature matrix (NumPy array of shape [n_samples, n_features]).
        
        Returns:
        - Predicted values (NumPy array of shape [n_samples]).
        """
        # Starting with the initial prediction
        preds = np.full(X.shape[0], self.initial_prediction)
        
        # Adding contributions from all trees
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        
        return preds


# Generating some synthetic data
# Creating a synthetic regression dataset
X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Gradient Boosting Tree model
model = GradientBoostingTree(n_estimators=200, learning_rate=0.1, max_depth=3, loss='squared_error')
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
