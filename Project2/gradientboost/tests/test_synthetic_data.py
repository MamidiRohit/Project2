import sys
import os
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Insert the project root directory into the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gradientboost.models.GradientBoost import GradientBoostModel

# Generate synthetic data for testing
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
gbr = GradientBoostModel(n_estimators=200, learning_rate=0.1, max_depth=3)
gbr.fit(X_train, y_train)

# Predictions on the test set
predictions = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
std_y = np.std(y_test)  # Standard deviation of the target variable

# Determine if MSE is "good"
threshold = std_y**2  # Variance of the target
is_mse_good = mse < threshold

# Display results
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
print("Standard Deviation of Target (y):", std_y)
print("Threshold for Good MSE (Variance of y):", threshold)
print("Is MSE Good?:", "Yes" if is_mse_good else "No")
print("Sample Predictions:", predictions[:10])
print("Sample True Values:", y_test[:10])
