# Gradient-Boosting Tree Algorithm Implementation:

# Importing neccessary libraries:

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array
from sklearn.datasets import make_regression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Preparing the model:

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_pred = None
    
    def fit(self, X, y):
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype="numeric")

        # Initialize the initial prediction and residuals
        self.init_pred = np.mean(y)
        residuals = y - self.init_pred
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)
            
            # Update residuals
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
    
    def predict(self, X):
        # Validate input features
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype="numeric")

        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.models:
            pred += self.learning_rate * tree.predict(X)
        return pred

# Generate synthetic data to test our model:

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the custom GBT model

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics

mse_syntheticdata = mean_squared_error(y_test, y_pred)
rmse_syntheticdata = np.sqrt(mse_syntheticdata)
r2_syntheticdata = r2_score(y_test, y_pred)

print("\nOur Model Performance Metrics:")
print(f"Mean Squared Error: {mse_syntheticdata:.4f}")
print(f"Root Mean Squared Error: {rmse_syntheticdata:.4f}")
print(f"R² Score: {r2_syntheticdata:.4f} \n")

# Training data evaluation
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

# Test data evaluation
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Training MSE: {train_mse}, R²: {train_r2}")
print(f"Test MSE: {test_mse}, R²: {test_r2} \n")

# Usage example with parameters

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Predictions:{predictions}")

# Plot 1: Actual vs Predicted Values

plt.subplot(1, 1, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Plot 2: Residuals Plot

residuals = y_test - y_pred
plt.subplot(1, 1, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Plot 3: Distribution of Residuals

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Distribution of Residuals')
plt.show()

## Load and prepare the California Housing dataset to test our model:

housing = fetch_california_housing()
X, y = housing.data, housing.target

print("Dataset Information:")
print("Features:", housing.feature_names)
print("Shape:", X.shape)
print("Target Variable: Median House Value")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train the model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse_HousingData = mean_squared_error(y_test, y_pred)
rmse_HousingData = np.sqrt(mse_HousingData)
r2_HousingData = r2_score(y_test, y_pred)

print("\nOur Model Performance Metrics:")
print(f"Mean Squared Error: {mse_HousingData:.4f}")
print(f"Root Mean Squared Error: {rmse_HousingData:.4f}")
print(f"R² Score: {r2_HousingData:.4f}")

# 1. Actual vs Predicted Values:

plt.subplot(1, 1, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Values')
plt.ylabel('Predicted House Values')
plt.title('Actual vs Predicted House Values')

# 2. Residuals Plot:

residuals = y_test - y_pred
plt.subplot(1, 1, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted House Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

# 3. Distribution of Residuals:

plt.subplot(1, 1, 1)
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Distribution of Residuals')

# 4. Feature Importance Plot:

feature_importance = np.zeros(X.shape[1])
for tree in model.models:
    feature_importance += tree.feature_importances_
feature_importance /= len(model.models)

plt.subplot(1, 1, 1)
plt.barh(range(len(housing.feature_names)), feature_importance)
plt.yticks(range(len(housing.feature_names)), housing.feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Housing Price Prediction')

plt.tight_layout()
plt.show()

# Cross-validation analysis:
# Comparing with sklearn's implementation

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR

sklearn_model = SklearnGBR(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
cv_scores = cross_val_score(sklearn_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)

print("\nCross-validation Results (RMSE):")
print(f"Mean RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")