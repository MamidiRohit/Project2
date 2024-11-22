# advanced_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from model_selection import k_fold_cross_validation, bootstrapping

# Set style for better visualizations
plt.style.use('tableau-colorblind10')

# Generate synthetic data with more features
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# Initialize model
model = LinearRegression()
model.fit(X, y)

# 1. Enhanced Regression Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='#1f77b4', alpha=0.5, label='Data points')
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Add confidence intervals
y_std = np.std(y - model.predict(X))
plt.plot(X_test, y_pred, color='#d62728', label='Regression line', linewidth=2)
plt.fill_between(X_test.flatten(), 
                 y_pred - 2*y_std, 
                 y_pred + 2*y_std, 
                 alpha=0.2, 
                 color='#d62728', 
                 label='95% Confidence interval')
plt.title('Linear Regression with Confidence Intervals')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. K-fold Cross-validation Analysis
k_values = [3, 5, 7, 10]
cv_errors = []

for k in k_values:
    error = k_fold_cross_validation(model, X, y, k=k, random_seed=42)
    cv_errors.append(error)

plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_errors, marker='o', linestyle='-', 
         color='#2ca02c', linewidth=2, markersize=8)
plt.title('Cross-validation Error vs K-folds')
plt.xlabel('Number of folds')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

# 3. Bootstrap Distribution Plot
n_iterations = 100
bootstrap_errors = []

for _ in range(n_iterations):
    error = bootstrapping(model, X, y, n_iterations=1, test_size=0.3, random_seed=None)
    if error is not None:
        bootstrap_errors.append(error)

plt.figure(figsize=(8, 5))
plt.hist(bootstrap_errors, bins=20, density=True, alpha=0.7, 
         color='#ff7f0e', edgecolor='black')
plt.axvline(np.mean(bootstrap_errors), color='#d62728', linestyle='--', 
            label=f'Mean MSE: {np.mean(bootstrap_errors):.3f}')
plt.title('Bootstrap Error Distribution')
plt.xlabel('Mean Squared Error')
plt.ylabel('Density')
plt.legend()

# 4. Residual Analysis
y_pred_all = model.predict(X)
residuals = y - y_pred_all

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_all, residuals, alpha=0.5, color='#9467bd')
plt.axhline(y=0, color='#d62728', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

# 5. Learning Curve
train_sizes = np.linspace(0.1, 1.0, 10)
train_errors = []
val_errors = []

for size in train_sizes:
    n_samples = int(len(X) * size)
    X_subset = X[:n_samples]
    y_subset = y[:n_samples]
    train_error = k_fold_cross_validation(model, X_subset, y_subset, k=5)
    val_error = k_fold_cross_validation(model, X_subset, y_subset, k=3)
    train_errors.append(train_error)
    val_errors.append(val_error)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes * 100, train_errors, label='Training Error', 
         color='#1f77b4', linewidth=2)
plt.plot(train_sizes * 100, val_errors, label='Validation Error', 
         color='#d62728', linewidth=2)
plt.title('Learning Curve')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()