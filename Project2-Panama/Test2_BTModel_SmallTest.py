import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from time import time
from boosting_tree.BoostingTreeModel import BoostingTreeModel, MyRSquared, MyMSE

# Load the dataset
dataset_path = 'small_test.csv'
data = pd.read_csv(dataset_path)

# Split into features (X) and target (y)
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values   # Last column

# Parameters for the model
num_trees = 50
learning_rate = 0.1
max_depth = 3
subsample = 0.8  # Added subsample for comparison

# Initialize and fit the custom Boosting Tree model
custom_model = BoostingTreeModel(num_trees=num_trees, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample)
start_time = time()
custom_results = custom_model.fit(X, y)
y_pred_custom = custom_results.predict(X)
custom_time = time() - start_time

# Initialize and fit the scikit-learn GradientBoostingRegressor for comparison
sklearn_model = GradientBoostingRegressor(
    n_estimators=num_trees,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    random_state=42
)
start_time = time()
sklearn_model.fit(X, y)
y_pred_sklearn = sklearn_model.predict(X)
sklearn_time = time() - start_time

# Calculate evaluation metrics
r_squared_custom = r2_score(y, y_pred_custom)
mse_custom = mean_squared_error(y, y_pred_custom)
r_squared_sklearn = r2_score(y, y_pred_sklearn)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

# Print comparison results
print("Test - Dataset from small_test.csv")
print(f"Custom Boosting Tree - R²: {r_squared_custom:.4f}, MSE: {mse_custom:.4f}, Time: {custom_time:.4f}s")
print(f"scikit-learn Boosting Tree - R²: {r_squared_sklearn:.4f}, MSE: {mse_sklearn:.4f}, Time: {sklearn_time:.4f}s")

# Q-Q Plot function for residual analysis
def plot_qq(y_true, y_pred_custom, y_pred_sklearn, test_name):
    plt.figure(figsize=(12, 6))

    # Residuals for both models
    residuals_custom = y_true - y_pred_custom
    residuals_sklearn = y_true - y_pred_sklearn

    # Q-Q plot for custom Boosting Tree model residuals
    plt.subplot(1, 2, 1)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals_custom, dist="norm")
    plt.scatter(osm, osr, color="blue", label="Data Points")
    plt.plot(osm, slope * osm + intercept, color="orange", label="Fit Line")
    plt.title(f'{test_name} - Q-Q Plot (Custom BoostingTree)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.legend()

    # Q-Q plot for Scikit-learn Boosting Tree model residuals
    plt.subplot(1, 2, 2)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals_sklearn, dist="norm")
    plt.scatter(osm, osr, color="purple", label="Data Points")
    plt.plot(osm, slope * osm + intercept, color="green", label="Fit Line")
    plt.title(f'{test_name} - Q-Q Plot (Scikit-learn Boosting)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Generate Q-Q plots
plot_qq(y, y_pred_custom, y_pred_sklearn, "Dataset Test")