# -*- coding: utf-8 -*-
"""ML Project-1.ipynb
Group Members:
A20584318 - ANSH KAUSHIK
A20593046 - ARUNESHWARAN SIVAKUMAR
A20588339 - HARISH NAMASIVAYAM MUTHUSWAMY
A20579993 - SHARANYA MISHRA
"""
import numpy as np

# Gradient Boosting Regressor
# Function to split data into training and testing sets (75% training, 25% testing)
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

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1):
        """
        n_estimators: Number of weak learners (trees)
        learning_rate: Step size for updating predictions
        max_depth: Maximum depth of individual decision stumps
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []  # List to store weak learners
        self.initial_prediction = None  # Mean of target values

    def _split(self, X, y):
        """
        Splits data based on feature threshold for decision stump.
        """
        best_feature, best_threshold, best_loss = None, None, float('inf')
        best_left, best_right = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_mean = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_mean = np.mean(y[right_mask]) if np.any(right_mask) else 0

                loss = (
                    np.sum((y[left_mask] - left_mean) ** 2) +
                    np.sum((y[right_mask] - right_mean) ** 2)
                )

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left_mask
                    best_right = right_mask

        return best_feature, best_threshold, best_left, best_right

    def fit(self, X, y):
        """
        Fits the Gradient Boosting Regressor.
        """
        # Initialize predictions to the mean of the target values
        self.initial_prediction = np.mean(y)
        predictions = np.full(y.shape, self.initial_prediction)

        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions

            # Fit a decision stump to the residuals
            feature, threshold, left_mask, right_mask = self._split(X, residuals)

            if feature is None:
                break

            model = {
                'feature': feature,
                'threshold': threshold,
                'left_value': np.mean(residuals[left_mask]),
                'right_value': np.mean(residuals[right_mask]),
            }

            # Update predictions
            predictions[left_mask] += self.learning_rate * model['left_value']
            predictions[right_mask] += self.learning_rate * model['right_value']

            # Save the weak learner
            self.models.append(model)

            # Print progress
            if i % 10 == 0:
                mse = np.mean((y - predictions) ** 2)
                print(f"Iteration {i}: MSE: {mse}")

    def predict(self, X):
        """
        Predicts using the fitted Gradient Boosting Regressor.
        """
        predictions = np.full(X.shape[0], self.initial_prediction)

        for model in self.models:
            feature = model['feature']
            threshold = model['threshold']
            left_value = model['left_value']
            right_value = model['right_value']

            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            predictions[left_mask] += self.learning_rate * left_value
            predictions[right_mask] += self.learning_rate * right_value

        return predictions

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)
    
    def f1_score(y_true, y_pred):
    # Calculate True Positives, False Positives, True Negatives, and False Negatives
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
    
        # Calculate Precision and Recall
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
    
        # Calculate F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return f1

    def gini_coefficient(y_true, y_pred):
        """Calculate the Gini coefficient."""
        sorted_indices = np.argsort(y_pred)
        sorted_true = y_true[sorted_indices]
        cum_true = np.cumsum(sorted_true) / np.sum(sorted_true)
        cum_population = np.cumsum(np.ones_like(sorted_true)) / len(sorted_true)
        gini = 1 - 2 * np.trapz(cum_true, cum_population)
        return gini