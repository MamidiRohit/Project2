import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')

print("relative_path", relative_path)
sys.path.append(relative_path)

from gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("For some of test cases it can take a while .....")
# Generate a large dataset with many features
np.random.seed(42)
n_samples = 400  # Large number of samples
n_features = 8   # Large number of features
X_large = np.random.rand(n_samples, n_features)
coefficients = np.random.randn(n_features)
y_large = X_large @ coefficients + np.random.randn(n_samples) * 0.1  # Linear combination with noise

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

# Initialize and fit the custom model
start_time = time.time()
gb = GradientBoosting(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',  # To handle high dimensionality
    handle_missing='none'
)
gb.fit(X_train, y_train)
custom_time = time.time() - start_time

# Make predictions with the custom model
y_pred = gb.predict(X_test)
gb.plot_predictions_vs_actual(y_pred, y_test)

# Initialize and fit scikit-learn's model
start_time = time.time()
sklearn_gb = SklearnGBR(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt'
)
sklearn_gb.fit(X_train, y_train)
sklearn_time = time.time() - start_time

# Make predictions with scikit-learn's model
y_pred_sklearn = sklearn_gb.predict(X_test)

# Evaluate the custom model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Evaluate scikit-learn's model
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)


print("Results for Large Dataset with Many Features:")
print("Custom Gradient Boosting Regressor Metrics:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R^2 Score: {r2:.4f}")
print(f"  Training Time: {custom_time:.2f} seconds")
print("Scikit-Learn Gradient Boosting Regressor Metrics:")
print(f"  MSE: {mse_sklearn:.4f}")
print(f"  MAE: {mae_sklearn:.4f}")
print(f"  R^2 Score: {r2_sklearn:.4f}")
print(f"  Training Time: {sklearn_time:.2f} seconds")

gb.plot_learning_curve("large dataset")