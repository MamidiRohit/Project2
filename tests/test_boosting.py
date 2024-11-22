import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from boosting.gradient_boosting import GradientBoostingTree

def test_gradient_boosting_training():
    ...




def test_gradient_boosting_training():
    """
    Test training of the GradientBoostingTree model.
    """
    # Create a synthetic dataset
    X = np.random.rand(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)

    # Initialize and train the model
    model = GradientBoostingTree(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    # Verify that the model has stored estimators
    assert len(model.models) == 10, "Number of estimators should match n_estimators."
    print("GradientBoostingTree training test passed.")

def test_gradient_boosting_prediction():
    """
    Test prediction of the GradientBoostingTree model.
    """
    # Create a synthetic dataset
    X_train = np.random.rand(100, 2)
    y_train = 4 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.normal(0, 0.1, 100)
    X_test = np.random.rand(20, 2)
    y_test = 4 * X_test[:, 0] + 3 * X_test[:, 1] + np.random.normal(0, 0.1, 20)

    # Initialize and train the model
    model = GradientBoostingTree(n_estimators=20, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Verify predictions have the same length as test data
    assert len(y_pred) == len(y_test), "Predictions should have the same length as test data."
    # Verify predictions are finite
    assert np.all(np.isfinite(y_pred)), "Predictions should not contain NaN or infinite values."
    print("GradientBoostingTree prediction test passed.")

def test_gradient_boosting_regression_performance():
    """
    Test the performance of the GradientBoostingTree model on a simple regression task.
    """
    # Create a synthetic dataset
    X = np.random.rand(200, 1)
    y = 5 * X[:, 0] + np.random.normal(0, 0.2, 200)

    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize and train the model
    model = GradientBoostingTree(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute MSE
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"GradientBoostingTree regression MSE: {mse:.4f}")

    # Verify MSE is reasonably small
    assert mse < 0.1, "MSE should be small for a simple linear relationship."
    print("GradientBoostingTree regression performance test passed.")

if __name__ == "__main__":
    print("Testing GradientBoostingTree training...")
    test_gradient_boosting_training()
    
    print("\nTesting GradientBoostingTree prediction...")
    test_gradient_boosting_prediction()
    
    print("\nTesting GradientBoostingTree regression performance...")
    test_gradient_boosting_regression_performance()
