# Gradient Boosting Trees:

## Team Members:
Name: Gurjot Singh Kalsi     
CWID: A20550984

Name: Siva Vamsi Kolli   
CWID: A20560901

Name: Sai Teja Reddy Janga     
CWID: A20554588

## Model Description:
Gradient Boosting Trees (GBT) are an ensemble learning method that builds a sequence of decision trees to improve predictive accuracy. The key idea behind GBT is to train decision trees iteratively, where each new tree attempts to correct the errors (residuals) made by the previous trees. By combining the predictions of many weak learners (shallow decision trees), GBT can create a strong predictive model.

### How Gradient Boosting Works:
1. The algorithm starts with an initial model that predicts a constant value (usually the mean of the target variable for regression).
2. At each iteration, the algorithm:
   - Calculates the residuals, which are the differences between the true values and the current model’s predictions.
   - Fits a new decision tree to the residuals.
   - Updates the model by adding the scaled predictions from the new tree.
3. The final model is the sum of the initial prediction and the predictions from all the trees.

### When to Use Gradient Boosting Trees:
- GBT is highly effective when there are complex, non-linear relationships between the input features and the target variable.
- It is suitable for both regression and classification problems.
- GBT works well with structured/tabular data, especially when feature interactions are not straightforward.
- It’s best used when you have a relatively large amount of data, as boosting models can be prone to overfitting on small datasets.

--------------

## Testing:
To ensure the correct implementation of the Gradient Boosting Tree algorithm, we conducted the following tests:

1. **Synthetic Data Testing**:
   - We generated synthetic regression data using Scikit-Learn’s `make_regression` function.
   - We trained the custom Gradient Boosting model on this data and evaluated its performance using the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R^2 Score (R^2) metrics.
   - The predictions from our model were compared with those from Scikit-Learn’s `GradientBoostingRegressor` to verify that our implementation produces similar results.

2. **Real-World Dataset**:
   - We tested the model on a publicly available dataset (e.g., the California Housing dataset).
   - We split the data into training and testing sets, then evaluated the model’s accuracy and MSE, RMSE, R^2 scores to confirm its generalization capability.

3. **Edge Cases**:
   - We tested the model on datasets with varying numbers of features and sample sizes to check for robustness.
   - We examined how the model behaves when trained on datasets with extreme outliers or high noise levels.

--------------

## Exposed Parameters:
Our implementation provides several parameters that users can adjust to fine-tune the model’s performance:

1. **n_estimators (default: 100)**:
   - The number of boosting rounds or decision trees to be built.
   - Increasing `n_estimators` improves model performance but also increases the risk of overfitting.

2. **learning_rate (default: 0.1)**:
   - A scaling factor that adjusts the contribution of each tree to the overall model.
   - Lower learning rates require more trees to achieve the same level of performance but can lead to better generalization.

3. **max_depth (default: 3)**:
   - The maximum depth of each decision tree.
   - Limiting the depth prevents the trees from overfitting and reduces the model’s complexity.

### Usage Example:
```python
from gradient_boosting import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Challenges:
While developing the Gradient Boosting Tree model, we encountered several challenges:

1. **Overfitting**:
   - GBT models are prone to overfitting, especially when using too many trees or deep trees.
   - This issue can be mitigated by tuning the `learning_rate`, `max_depth`, and `n_estimators` parameters, as well as implementing techniques like **early stopping** and **tree pruning**.

2. **High Computational Cost**:
   - Training GBT models can be computationally expensive, especially with large datasets or a high number of trees.
   - Optimizing the implementation (e.g., using parallel processing or GPU acceleration) could significantly speed up training.

3. **Handling Noisy Data**:
   - GBT can be sensitive to noisy data, leading to poor generalization if the noise is not handled properly.
   - Techniques such as regularization, limiting tree depth, and using robust loss functions can help alleviate this issue.

4. **Edge Cases**:
   - The current implementation might struggle with datasets containing extreme outliers, as the decision trees may fit too closely to these anomalies.
   - Given more time, implementing regularization techniques and more robust loss functions could address these edge cases.

--------------

## Conclusion:
This project involved implementing a custom Gradient Boosting Tree model from scratch and testing it on various datasets. The model performs well on structured data with complex relationships and can be fine-tuned using several hyperparameters. However, there are some inherent challenges, such as the risk of overfitting and computational inefficiency, which could be addressed with further optimizations.

### Potential Future Enhancements:
- Implement **early stopping** to prevent overfitting.
- Add support for **classification tasks** to broaden the model’s applicability.
- Optimize training using **parallel processing** or **GPU acceleration** to improve computational efficiency.
- Introduce **regularization techniques** like tree pruning and limiting the number of leaf nodes to enhance model robustness.
