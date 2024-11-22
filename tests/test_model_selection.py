
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from boosting.gradient_boosting import GradientBoostingTree

def test_gradient_boosting_training():
    ...

from model_selection.cross_validation import k_fold_cv
from model_selection.bootstrapping import bootstrap
from boosting.gradient_boosting import GradientBoostingTree

def test_k_fold_cv():
    """
    Test k-fold cross-validation.
    """
    # Create synthetic dataset
    X = np.random.rand(100, 2)
    y = 4 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, size=100)

    # Initialize the Gradient Boosting model
    model = GradientBoostingTree(n_estimators=10, learning_rate=0.1, max_depth=3)

    # Perform k-fold cross-validation
    score = k_fold_cv(model, X, y, k=5, metric="mse")
    
    assert score > 0, "Cross-validation score should be greater than 0."
    print(f"K-Fold Cross-Validation MSE: {score:.4f}")

def test_bootstrap():
    """
    Test bootstrapping for model evaluation.
    """
    # Create synthetic dataset
    X = np.random.rand(100, 2)
    y = 4 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, size=100)

    # Initialize the Gradient Boosting model
    model = GradientBoostingTree(n_estimators=10, learning_rate=0.1, max_depth=3)

    # Perform bootstrapping
    scores, mean_score = bootstrap(model, X, y, B=10, metric="mse")
    
    assert len(scores) == 10, "Number of bootstrap scores should match the number of iterations."
    assert mean_score > 0, "Mean bootstrap score should be greater than 0."
    print(f"Bootstrap Mean MSE: {mean_score:.4f}")

if __name__ == "__main__":
    print("Testing K-Fold Cross-Validation...")
    test_k_fold_cv()
    
    print("\nTesting Bootstrapping...")
    test_bootstrap()
