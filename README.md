## Boosting Tree Model Documentation

**Student Name and A#**

Harlee Ramos A20528450

Andres Orozco A20528634

**Group Name: CS584-Project2-Panama**

This document provides two sections:

1. Instructions on how to run the Boosting Tree model.
2. Answers to Project 2 questions. (This section is likely not included in the provided code)

---

## Instructions on How to Use the Boosting Tree Model

**Importing the Model**

To use the Boosting Tree Model, import the required classes:

```
from boosting_tree import BoostingTreeModel, MyMSE, MyRSquared, DecisionTreeRegressorCustom
```

---

**Training the Model**

To train the Boosting Tree model, use the `fit` method. It requires:

* **X**: A 2D NumPy array representing the feature matrix.
* **y**: A 1D NumPy array representing the target labels.

**Example:**

```python
import numpy as np
from boosting_tree import BoostingTreeModel

# Generate example data
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 100)

# Initialize the model
model = BoostingTreeModel(num_trees=20, learning_rate=0.1, max_depth=3, tol=1e-5, subsample=0.8)

# Train the model
results = model.fit(X, y)
```

---

**Making Predictions**
To make predictions with the Boosting Tree model, use the `predict` method:
The predict method makes predictions for new data points. It takes the following argument:
* **X**: A 2D NumPy array with the same number of features as the training data.

```python
# Example usage
new_data = np.array([[0.2, 0.4, 0.1]])
predictions = model.predict(new_data)
print(predictions)
```


**Evaluation with Metrics**
To evaluate model performance, use the MyMSE and MyRSquared classes. Both classes provide static methods for calculating metrics.
Example:
```python
from boosting_tree import MyMSE, MyRSquared

# Calculate Mean Squared Error
mse = MyMSE.calculate(y, y_pred)
print("Mean Squared Error:", mse)

# Calculate R-squared
r_squared = MyRSquared.calculate(y, y_pred)
print("R-squared:", r_squared)
```

Sample runnable script:
```python
import numpy as np
from boosting_tree import BoostingTreeModel, MyMSE, MyRSquared

# Sample Data
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + X[:, 1] - 1.5 * X[:, 2] + np.random.normal(0, 0.05, 100)

# Model Initialization
model = BoostingTreeModel(num_trees=50, learning_rate=0.05, max_depth=4, tol=1e-4, subsample=0.7)

# Train the Model
results = model.fit(X, y)

# Predict
y_pred = results.predict(X)

# Evaluate
mse = MyMSE.calculate(y, y_pred)
r_squared = MyRSquared.calculate(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r_squared:.4f}")
```

**Additional Details**

**Key Classes and Functions**

| Class/Function | Description |
|---|---|
| `BoostingTreeModel` | Main class for training and predicting using gradient boosting. |
| `MyMSE` | Static class for calculating Mean Squared Error. |
| `MyRSquared` | Static class for calculating R-squared. |
| `DecisionTreeRegressorCustom` | Custom decision tree regressor used as the base model. |

**Parameters for `BoostingTreeModel`**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_trees` | int | 20 | Number of boosting iterations (trees). |
| `learning_rate` | float | 0.1 | Step size for updates. |
| `max_depth` | int | 3 | Maximum depth of each decision tree. |
| `tol` | float | 1e-5 | Tolerance for early stopping. |
| `subsample` | float | 0.5 | Fraction of samples used for training. |
