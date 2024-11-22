"""
This module provides utility functions for evaluating and tuning the Gradient Boosting Tree model.

Functions:
----------
1. r2_score_manual(y_true, y_pred):
    Computes the R² (coefficient of determination) score manually, measuring how well the predictions match the true values.

    Parameters:
    - y_true: array-like
        The ground truth target values.
    - y_pred: array-like
        The predicted target values.

    Returns:
    - r2: float
        The R² score, where 1 indicates perfect predictions and values close to 0 indicate poor predictions.

2. grid_search_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators_values, learning_rate_values, max_depth_values):
    Performs grid search to find the best hyperparameters for a Gradient Boosting Tree model.

    Parameters:
    - X_train: array-like
        Training feature set.
    - y_train: array-like
        Training target values.
    - X_test: array-like
        Testing feature set.
    - y_test: array-like
        Testing target values.
    - n_estimators_values: list
        A list of integers representing the number of trees (n_estimators) to evaluate.
    - learning_rate_values: list
        A list of floats representing the learning rates to evaluate.
    - max_depth_values: list
        A list of integers representing the maximum depth of trees to evaluate.

    Returns:
    - best_params: dict
        A dictionary containing the best hyperparameter values for `n_estimators`, `learning_rate`, and `max_depth`.
    - best_score: float
        The highest R² score obtained during the grid search.
"""

import numpy as np
from models.GradientBoost import GradientBoostingTree

def r2_score_manual(self, y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (rss / tss)
    return r2


def grid_search_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators_values, learning_rate_values, max_depth_values):
    best_score = -float("inf")
    best_params = None

    for n_estimators in n_estimators_values:
        for learning_rate in learning_rate_values:
            for max_depth in max_depth_values:
                
                model = GradientBoostingTree(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                
                score = model.r2_score_manual(y_test, y_pred)

                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                    }

    return best_params, best_score