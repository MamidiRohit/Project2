# main.py

import numpy as np
from model_selection import k_fold_cross_validation, bootstrapping
from linear_regression import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

#Choose any random model
model = LinearRegression()

# Perform 5-fold cross-validation
kfold_error = k_fold_cross_validation(model, X, y, k=5, random_seed=42)
print("5-Fold Cross-Validation Error (MSE):", kfold_error)

# Perform bootstrap with 100 iterations
bootstrap_error = bootstrapping(model, X, y, n_iterations=100, test_size=0.33, random_seed=42)
print("Bootstrap Error (MSE):", bootstrap_error)
