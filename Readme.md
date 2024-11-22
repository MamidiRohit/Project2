# Project 2 
BY

A20539949-Usha Devaraju

A20548244-Roopashri Kommana

A20550565-Sai Sandeep Neerukonda

# Gradient Boosting for Regression

This repository contains a custom implementation of a Gradient Boosting model for regression tasks, using decision trees as base learners. The model is designed to be versatile and easily adjustable to fit various regression problems.

# 1.What does the model you have implemented do and when should it be used?
## Model Description

The Gradient Boosting model implemented here constructs an ensemble of decision trees in a sequential manner, where each tree is built to correct the errors made by the previous ones. The model is particularly useful for datasets where relationships between features and the target variable are complex and non-linear.

### When to Use This Model

This model should be used when:
- Dealing with regression tasks requiring robust predictive power.
- Handling datasets with complex and non-linear relationships.
- Situations where other simpler models (like linear regression) are insufficient.

# 2.How did you test your model to determine if it is working reasonably correctly?
## Testing the Model

The model has been rigorously tested using the California Housing dataset, which is a standard dataset for evaluating regression models. The testing involves:
- Splitting the data into training and testing sets.
- Scaling the feature matrix to standardize the input data.
- Training the Gradient Boosting model on the training data.
- Evaluating its performance using Mean Squared Error (MSE) on the test set.

# 3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.
## Exposed Parameters

Users can tune the following parameters to optimize the model's performance:
- `n_estimators`: The number of trees to build (default is 100).
- `learning_rate`: The step size at each iteration to control overfitting (default is 0.1).
- `max_depth`: The maximum depth of each decision tree (default is 3).

### Prerequisites

Ensure you have Python installed along with the following libraries:
- `numpy`
- `scikit-learn`

To install missing dependencies, use:
```bash
pip install numpy scikit-learn
```

### Basic Usage Example

```python
from gradient_boosting import GradientBoosting
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the gradient boosting model
model = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
predictions = model.predict(X_test_scaled)
print(predictions)
```

## Running Tests
To test the model on the California Housing dataset, run:
```python
python testing.py
```
The script will:
- **Load the dataset.**
- **Train and test the Gradient Boosting model.**
- **Output the Mean Squared Error (MSE) of the predictions.**

# 4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
## Potential Issues and Workarounds

The model may encounter difficulties with specific types of inputs such as:

- **Extremely Noisy Data**: High levels of noise can lead to overfitting, where the model learns the noise as patterns, degrading prediction accuracy on new data.
- **Outliers**: Outliers can disproportionately influence the decision boundaries established by the decision trees, leading to suboptimal models.

### Workarounds

To enhance model robustness and performance:
- **Preprocessing Steps**: Implement robust preprocessing steps to handle outliers and noise, such as outlier detection algorithms or robust scaling methods.
- **Advanced Techniques**: Explore integrating outlier detection algorithms and advanced noise filtering techniques before fitting the model to improve its generalization capabilities.

