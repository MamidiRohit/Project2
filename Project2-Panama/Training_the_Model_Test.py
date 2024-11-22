import numpy as np
from boosting_tree.BoostingTreeModel import BoostingTreeModel

# Generate example data
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

# Initialize the model
model = BoostingTreeModel(num_trees=20, learning_rate=0.1, max_depth=3, tol=1e-5, subsample=0.8)

# Train the model
results = model.fit(X, y)


# Example usage
new_data = np.array([[0.2, 0.4, 0.1]])
predictions = model.predict(new_data)
print(predictions)
