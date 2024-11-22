import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')

print("relative_path", relative_path)
sys.path.append(relative_path)


from gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("For some of test cases it can take a while .....")

# Generate a dataset with missing values
np.random.seed(42)
n_samples = 100
n_features = 5
X_missing = np.random.rand(n_samples, n_features)
y_missing = np.sin(X_missing[:, 0]) + np.cos(X_missing[:, 1]) + np.random.randn(n_samples) * 0.1

# Introduce missing values randomly
missing_mask = np.random.rand(n_samples, n_features) < 0.1  # 10% missing values
X_missing[missing_mask] = np.nan

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.2, random_state=42)

# Initialize and fit the custom model with missing value handling
gb = GradientBoosting(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    handle_missing='mean'  # Options: 'none', 'mean', 'median'
)
gb.fit(X_train, y_train)

# Make predictions with the custom model
y_pred = gb.predict(X_test)
# Initialize and fit scikit-learn's model
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

sklearn_gb = SklearnGBR(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None
)
sklearn_gb.fit(X_train_imputed, y_train)

# Make predictions with scikit-learn's model
y_pred_sklearn = sklearn_gb.predict(X_test_imputed)

# Evaluate the custom model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Evaluate scikit-learn's model
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("Results for Dataset with Missing Values:")
print("Custom Gradient Boosting Regressor Metrics:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R^2 Score: {r2:.4f}")
print("Scikit-Learn Gradient Boosting Regressor Metrics:")
print(f"  MSE: {mse_sklearn:.4f}")
print(f"  MAE: {mae_sklearn:.4f}")
print(f"  R^2 Score: {r2_sklearn:.4f}")

gb.plot_learning_curve("Missing values dataset")
