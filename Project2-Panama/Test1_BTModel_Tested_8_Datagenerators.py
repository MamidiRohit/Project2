import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from boosting_tree.BoostingTreeModel import BoostingTreeModel, MyRSquared, MyMSE
from data_generators.data_generators import (
    linear_data_generator1,
    linear_data_generator2,
    nonlinear_data_generator1,
    generate_collinear_data,
    generate_periodic_data,
    generate_higher_dim_data,
    generate_high_collinear_data,
    generate_horrible_data,
)
from time import time


def plot_qq(y_true, y_pred_custom, y_pred_sklearn, test_name):
    plt.figure(figsize=(10, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(y_true - y_pred_custom, dist="norm")
    plt.scatter(osm, osr, label="Custom Model", color="blue")
    plt.plot(osm, slope * osm + intercept, color="orange", label="Fit Line (Custom)")
    plt.title(f"{test_name} Q-Q Plot (Custom Model)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(y_true - y_pred_sklearn, dist="norm")
    plt.scatter(osm, osr, label="Sklearn Model", color="green")
    plt.plot(osm, slope * osm + intercept, color="purple", label="Fit Line (Sklearn)")
    plt.title(f"{test_name} Q-Q Plot (Sklearn Model)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_test(X, y, test_name):
    print(f"Running {test_name}...")

    # Custom Model
    custom_model = BoostingTreeModel(num_trees=50, learning_rate=0.1, max_depth=3)
    start_time = time()
    custom_model.fit(X, y)
    y_pred_custom = custom_model.predict(X)
    custom_time = time() - start_time

    # Scikit-learn Model
    sklearn_model = GradientBoostingRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
    )
    start_time = time()
    sklearn_model.fit(X, y)
    y_pred_sklearn = sklearn_model.predict(X)
    sklearn_time = time() - start_time

    # Metrics
    custom_r2 = MyRSquared.calculate(y, y_pred_custom)
    custom_mse = MyMSE.calculate(y, y_pred_custom)
    sklearn_r2 = r2_score(y, y_pred_sklearn)
    sklearn_mse = mean_squared_error(y, y_pred_sklearn)

    print(f"Custom Model - R²: {custom_r2:.4f}, MSE: {custom_mse:.4f}, Time: {custom_time:.4f}s")
    print(f"Sklearn Model - R²: {sklearn_r2:.4f}, MSE: {sklearn_mse:.4f}, Time: {sklearn_time:.4f}s")

    # Plot Results
    plt.figure(figsize=(12, 6))
    plt.scatter(y, y_pred_custom, label="Custom Model", color="blue", alpha=0.6)
    plt.scatter(y, y_pred_sklearn, label="Sklearn Model", color="green", alpha=0.6)
    plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="--", label="Ideal Fit")
    plt.title(f"{test_name} - Model Predictions")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Q-Q Plot
    plot_qq(y, y_pred_custom, y_pred_sklearn, test_name)


if __name__ == "__main__":
    # Test 1: Linear Data 1
    X, y = linear_data_generator1(2, 3, [-5, 5], 100, seed=42)
    run_test(X, y, "Test 1: Linear Data (Single Feature)")

    # Test 2: Linear Data 2
    X, y = linear_data_generator2([1.5, -2.0], 5, [-5, 5], 100, seed=42)
    run_test(X, y, "Test 2: Linear Data (Multiple Features)")

    # Test 3: Nonlinear Data
    X, y = nonlinear_data_generator1(0.5, 2, [-5, 5], 100, seed=42)
    run_test(X, y, "Test 3: Nonlinear Data")

    # Test 4: Collinear Data
    X, y = generate_collinear_data([-5, 5], 0.01, (100, 3), seed=42)
    run_test(X, y, "Test 4: Collinear Data")

    # Test 5: Periodic Data
    X, y = generate_periodic_data(5, 10, [-5, 5], 0.5, 100, seed=42)
    run_test(X, y, "Test 5: Periodic Data")

    # Test 6: Higher Dimensional Data
    X, y = generate_higher_dim_data([-5, 5], 0.5, (100, 3), seed=42)
    run_test(X, y, "Test 6: Higher Dimensional Data")

    # Test 7: High Collinearity Data
    X, y = generate_high_collinear_data(50, 100, seed=42)
    run_test(X, y, "Test 7: High Collinearity Data")

    # Test 8: Extreme Scenario Data
    X, y = generate_horrible_data(1000, 100, seed=42)
    run_test(X, y, "Test 8: Extreme Scenario Data")