# Gradient Boosting Trees:

## Team Members:
Name: Gurjot Singh Kalsi     
CWID: A20550984

Name: Siva Vamsi Kolli   
CWID: A20560901

Name: Sai Teja Reddy Janga     
CWID: A20554588

--------------

## Instructions to Run the Code:

### Prerequisites:
1. Install python in your code editor environment using the script "pip install python". To Install Jupyter in your Python environment use the script "pip install notebook pip install jupyter".
2. To install neccessary libraries run the script "pip install numpy sklearn matplotlib seaborn".

### Running the Code:
1. Clone the repository or download the code files.
2. Open Code editor Navigate to the project directory.
3. To run the python file: 
   - Run the script "python GBT_Implementation.py".
   - The code will automatically:
      - Generate and test on synthetic data.
      - Run experiments on California Housing dataset.
      - Display performance metrics.
      - Show visualization plots.
4. To run the jupyter file: 
   - Option 1: Run All
      - Click the "Run All" button at the top of the notebook Or press Ctrl+Alt+R (Windows/Linux) or Cmd+Alt+R (Mac).
   - Option 2: Run Individual Cells
      - Click the play button next to each cell Or press Shift+Enter when inside a cell.

--------------

## Model Description:
Gradient Boosting Trees (GBT) are an ensemble learning method that builds a sequence of decision trees to improve predictive accuracy. The key idea behind GBT is to train decision trees iteratively, where each new tree attempts to correct the errors (residuals) made by the previous trees. By combining the predictions of many weak learners (shallow decision trees), GBT can create a strong predictive model.

### How Gradient Boosting Works:
1. The algorithm starts with an initial model that predicts a constant value (the mean of the target variable as initial prediction for regression).
2. At each iteration, the algorithm Sequentially adds scaled predictions from each tree:
   - Calculates the residuals, which are the differences between the true values and the current model’s predictions.
   - Fits a new decision tree to the residuals.
   - Updates the model by adding the scaled predictions from the new tree.
3. The final model is the sum of the initial prediction and the predictions from all the trees.

Our GBT model makes predictions through the following process:

```python
def predict(self, X):
    pred = np.full(X.shape[0], self.init_pred)  # Start with mean prediction
    for tree in self.models:
        pred += self.learning_rate * tree.predict(X)  # Add each tree's contribution
    return pred
```

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
   - The model's performance on synthetic data demonstrates a good fit, as indicated by the following metrics:
      - Mean Squared Error (MSE): 1241.21 — This measures the average squared difference between predicted and actual values. Lower values indicate better model performance.
      - Root Mean Squared Error (RMSE): 35.23 — This is the square root of MSE, providing an interpretable measure of error in the same units as the target variable.
      - R² Score: 0.9264 — This indicates that approximately 92.64% of the variance in the synthetic data is explained by the model, showing strong predictive capability.
   - These metrics suggest the model effectively captures the underlying patterns in the synthetic data with minimal error.

2. **Real-World Dataset**:
   - We tested the model on a publicly available dataset (e.g., the California Housing dataset).
   - We split the data into training and testing sets, then evaluated the model’s accuracy and MSE, RMSE, R^2 scores to confirm its generalization capability.
   - The model's evaluation on the California housing dataset indicates the following performance:
      - Mean Squared Error (MSE): 0.2375 — The average squared difference between predicted and actual housing values, showing a relatively low level of error.
      - Root Mean Squared Error (RMSE): 0.4874 — The square root of the MSE, providing an interpretable error measure in the same units as the housing prices.
      - R² Score: 0.8187 — Approximately 81.87% of the variance in the housing prices is explained by the model, indicating a strong predictive        relationship but leaving room for improvement.
   - The model performs well on the California housing data, explaining a substantial portion of the variance with minimal prediction error. However, the lower R² score compared to synthetic data suggests the real-world data's complexity might require further tuning or additional features to improve accuracy.

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

1. Basic usage: 

```python
# Default parameters
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
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