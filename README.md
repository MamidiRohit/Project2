# Project 2

# Submitted by:
- **Sana Samad**: A20543001
- **Naveed Mohiuddin**: A20540067
- **Owaiz Majoriya**: A20546104

---

# Gradient Boosting Regressor Implementation

## Project Overview
This project implements a Gradient Boosting Regressor, which is an ensemble learning method for regression tasks. The model iteratively trains weak learners (decision trees) to minimize the residual errors of previous iterations. This implementation was built using Python and relies on `sklearn` for constructing the decision trees but handles the boosting logic manually.

## Features
- **Gradient Boosting Framework**: Utilizes iterative training of decision trees to optimize predictions.
- **Customizable Parameters**: Provides users with control over `n_estimators`, `learning_rate`, and `max_depth` to fine-tune model performance.
- **Performance Evaluation**: Includes evaluation using Mean Squared Error (MSE) and scatter plots for true vs. predicted values.

---

## Setup

### Prerequisites
Ensure you have the following Python libraries installed:
- Python 3.x
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

Install them via pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### How to Run
1. Clone the repository or download the files.
   ```bash
   git clone <your_repo_url>
   cd <your_project_directory>
   ```

2. Ensure your dataset is a CSV file in the expected format:
   - All columns except the last one are features.
   - The last column is the target variable.

3. Modify `evaluate_model.ipynb` to include the correct path to your dataset.

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook evaluate_model.ipynb
   ```

5. The notebook will preprocess the data, train the Gradient Boosting Regressor, evaluate it, and visualize the results.

---

## Parameters
You can adjust the following parameters for tuning performance:
- `n_estimators`: Number of trees in the ensemble.
- `learning_rate`: Controls the contribution of each tree.
- `max_depth`: Maximum depth of each decision tree.

---

## Outputs
- **Training Metrics**: MSE is displayed as a quantitative evaluation of the model's predictions.
- **Visualization**: A scatter plot compares true and predicted values to illustrate performance.

---

## How to Run the Example
Here is an example of training and evaluating the Gradient Boosting Regressor:

```python
from GradientBoostingTree import CustomGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Initialize the model
gbr = CustomGradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)

# Train the model
gbr.fit(X_train, y_train)

# Make predictions
y_pred = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
```

---

# QUESTIONS:

1. **What does the model do, and when should it be used?**
   The model implements Gradient Boosting Regression, a machine learning algorithm for predicting numerical targets. It iteratively trains decision trees to minimize residual errors. This model is ideal for:
   - Regression problems where high accuracy is needed.
   - Scenarios with non-linear relationships between features and targets.
   - Cases where overfitting can be controlled via boosting parameters.

2. **How was the model tested?**
   The model was tested using:
   - An 80-20 train-test split on the dataset.
   - Evaluation of predictions against the ground truth using Mean Squared Error (MSE).
   - Visualization of true vs. predicted values via scatter plots.

3. **Exposed Parameters for Tuning Performance:**
   The following parameters are exposed for tuning:
   - `n_estimators`: Number of boosting iterations (trees).
   - `learning_rate`: Contribution of each tree to the final prediction.
   - `max_depth`: Limits the depth of each decision tree, controlling overfitting.

4. **Troublesome Inputs and Possible Solutions:**
   The model may struggle with:
   - **High-dimensional data**: Training time increases significantly with many features.
     - *Solution*: Use dimensionality reduction techniques like PCA.
   - **Outliers**: Sensitive to outliers in the target variable.
     - *Solution*: Preprocess the data to remove or handle outliers.
   - **Imbalanced datasets**: May not handle heavily skewed data distributions well.
     - *Solution*: Use sampling techniques or modify the loss function.
   - **Large datasets**: Computationally intensive for large datasets.
     - *Solution*: Parallelize the implementation or use distributed frameworks.

---
