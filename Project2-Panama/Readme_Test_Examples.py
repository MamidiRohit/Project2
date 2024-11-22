import numpy as np
from boosting_tree.BoostingTreeModel import BoostingTreeModel, MyRSquared, MyMSE

# Sample Data
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + X[:, 1] - 1.5 * X[:, 2] + np.random.normal(0, 0.05, 100)

# Model Initialization
model = BoostingTreeModel(num_trees=50, learning_rate=0.05, max_depth=4, tol=1e-4, subsample=0.7)

# Train the Model
results = model.fit(X, y)

# Predict
y_pred = results.predict(X)

# Calculate Mean Squared Error
mse = MyMSE.calculate(y, y_pred)
# Calculate R-squared
r_squared = MyRSquared.calculate(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r_squared:.4f}")
