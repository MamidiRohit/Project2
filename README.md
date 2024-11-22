# Wine Quality Dataset - Custom Model Selection Techniques (K Fold)

Group Members Details:

- **Sai Pranay Yada** (A20553636)
- **Kevan Dedania** (A20522659)
- **Hemanth Vennelakanti** (A20526563)
- **Kiran Velamati** (A20525555)

This project showcases custom implementations of model selection techniques applied to a wine quality dataset. It includes methods such as **k-fold cross-validation**, **bootstrapping**, and **AIC** to evaluate and compare different linear regression models. The goal is to demonstrate hands-on implementation of machine learning algorithms and evaluation metrics for both regression and classification tasks, all while avoiding pre-built libraries like scikit-learn.

This project emphasizes a deeper understanding of the underlying principles of model selection and evaluation.

## Features

- **Linear Regression from Scratch**  
  Build and fit regression models without relying on external libraries.

- **k-Fold Cross-Validation and Bootstrapping**  
  Validate model performance using different techniques to avoid overfitting and underfitting.

- **Regression and Classification Metric Calculations**  
  Evaluate models using custom implementations of MSE, MAE, accuracy, etc.

- **Applicability to Different Datasets**  
  Methods are designed to work with datasets beyond the wine quality dataset.

## Implementation

### 1. **CustomLinearRegression**

A lightweight and simple implementation of linear regression, focusing on educational purposes to understand the underlying mathematics.

#### **Attributes**

- `self.coef_`: Stores the coefficients (weights) of the model.
- `self.intercept_`: Stores the intercept term of the model.

#### **Methods**

1. **`fit(X, y)`**

   - Adds a bias term (intercept) to the input features `X` by concatenating a column of ones.
   - Uses the **normal equation** to calculate the parameters:  
     \[
     \theta = (X^T X)^{-1} X^T y
     \]
     Here, \(\theta\) is the vector containing both coefficients and the intercept.
   - Separates and stores the intercept as `self.intercept_` and the remaining weights as `self.coef_`.

2. **`predict(X)`**
   - Predicts target values using the formula:  
     \[
     \hat{y} = X \cdot \text{coef\_} + \text{intercept\_}
     \]

### 2. **k-Fold Cross-Validation**

This function evaluates a model's performance by splitting the dataset into **k folds** and iteratively training and testing on these folds. It ensures that each part of the dataset gets a chance to be used for both training and validation.

#### **Parameters**

- `X, y`: Input features and labels.
- `model`: The model to train and evaluate (e.g., `CustomLinearRegression`).
- `k` (optional): Number of folds. Default is 5.

#### **Steps**

1. Shuffle the dataset indices to randomize the data order.
2. Divide the dataset into **k approximately equal parts (folds)**.
3. For each fold:
   - Use the fold as the **validation set**.
   - Use the remaining data as the **training set**.
   - Train the model on the training set.
   - Predict and compute **Mean Squared Error (MSE)** on the validation set.
4. Return the **mean** and **standard deviation** of the MSE across all folds.

### 3. **Bootstrapping for Model Selection**

This function evaluates a model's performance by creating multiple **resampled datasets** using the bootstrap sampling method. It helps estimate the stability and variability of the model's performance.

#### **Parameters**

- `X, y`: Input features and labels.
- `model`: The model to train and evaluate (e.g., `CustomLinearRegression`).
- `n_iterations`: Number of bootstrap iterations to perform.

#### **Steps**

1. For each iteration:
   - Create a **resampled dataset** by sampling from the original data **with replacement**.
   - Train the model on the resampled data.
   - Evaluate **Mean Squared Error (MSE)** on the **original dataset**.
2. Compute and return the **mean** and **standard deviation** of the MSE across all iterations.

## 4. **Custom k-Fold Random Forest Implementation**

This implementation combines the custom Random Forest model with the k-Fold cross-validation technique to evaluate the model's performance. It ensures that the dataset is split into distinct training and testing sets across multiple iterations for robust evaluation.

---

### **Inputs**

- `X`: Features matrix (2D array or DataFrame).
- `y`: Target variable (1D array or Series).
- `model`: A Random Forest model instance (custom implementation).
- `k`: Number of folds for cross-validation (default is 5).

---

### **Steps**

#### 1. **Shuffle and Split the Data**

- Shuffle the dataset indices to ensure randomness.
- Divide the dataset into `k` approximately equal parts (folds).

#### 2. **Iterative Training and Testing**

For each fold:

- Use it as the **test set**.
- Combine the remaining folds into the **training set**.
- Train the Random Forest model on the training set.
- Predict on the test set using the trained model.
- Compute performance metrics (e.g., accuracy, Mean Squared Error (MSE)) for the fold.

#### 3. **Aggregate Metrics**

- Store the computed metrics for each fold.
- Calculate the **mean** and **standard deviation** of these metrics across all folds.

### **Outputs**

- Returns the **mean** and **standard deviation** of the evaluation metrics.

### **Algorithm**

#### **1. Initialization**

The Random Forest class is initialized with parameters:

- `n_trees`: Number of decision trees in the forest.
- `max_depth`: Maximum depth of each decision tree.
- `min_samples_split`: Minimum samples required to split a node.

#### **2. Training (`fit` Method)**

For each tree:

1. Create a **bootstrap sample** of the dataset.
2. Train a **decision tree** on this sample, selecting a random subset of features at each split.
3. Store the trained decision tree in the `trees` attribute.

#### **3. Prediction (`predict` Method)**

For a given input `X`:

1. Pass the data through all trained decision trees to obtain individual predictions.
2. Aggregate predictions:
   - For classification: Use **majority voting**.
   - For regression: Compute the **average**.

---

# Model Selection with Custom Implementations

This project showcases generic implementations of **k-Fold Cross-Validation** and **Bootstrapping** for model selection, applied to both regression and classification tasks. It emphasizes the development of custom algorithms and metrics, avoiding reliance on pre-built libraries like `scikit-learn`. The implementations are tested on popular datasets to demonstrate their effectiveness and robustness.

---

## **Note**

In this project, we implemented generic k-fold cross-validation and bootstrapping methods for model selection, showcasing their use in both regression and classification tasks. For regression, we developed a custom k-fold cross-validation approach, a custom linear regression model, and evaluation metrics to assess regression performance, applying these techniques to the Wine Dataset to predict wine quality.  
For classification, we implemented a custom k-fold cross-validation method, a custom Random Forest classifier, and evaluation metrics to evaluate classification performance, again using the Wine Dataset.

Additionally, to validate our implementations further, we tested the custom regression model on the Boston Housing dataset and the custom classification model on the Iris dataset, demonstrating their effectiveness and robustness.

---

## **Comparison of Model Selection Techniques**

### **Do Cross-Validation and Bootstrapping Model Selectors Agree with AIC in Simple Cases (e.g., Linear Regression)?**

Agreement:  
• In simple cases like linear regression, the custom cross-validation and bootstrapping selectors often align with AIC, provided the dataset fits the assumptions of linear regression (e.g., linearity, homoscedasticity).  
• All three methods evaluate the model's goodness of fit, but they focus on different aspects:  
• AIC evaluates the tradeoff between model complexity (number of parameters) and goodness of fit, favoring models that balance simplicity with fit.  
• Cross-validation focuses on how well the model generalizes to unseen data, ensuring the model performs well on data not seen during training.  
• Bootstrapping assesses how stable the model’s performance is by resampling the data multiple times and measuring variability in performance.

Disagreement:  
• The methods might disagree in cases of:  
• Nonlinear relationships that violate linear regression assumptions. AIC might still penalize the complexity of the model, while cross-validation and bootstrapping might reflect better or worse performance depending on the model's ability to generalize or overfit.  
• Small datasets where both cross-validation and bootstrapping can yield unstable results due to high variance in performance estimates.  
• High collinearity among predictors in linear regression, where AIC might favor simpler models that avoid overfitting, but cross-validation and bootstrapping may still indicate overfitting or instability with more predictors.

## **When Might the Methods Fail or Give Incorrect/Undesirable Results?**

**Cross-validation:**  
• Computationally expensive for large datasets: Since each fold requires retraining the model, this can be prohibitively slow for large datasets.  
• Biased results on imbalanced datasets or non-i.i.d. data (e.g., time series), where standard k-fold cross-validation might overestimate model performance due to improper data partitioning.

**Bootstrapping:**  
• May overfit the model to specific patterns in the data, especially when there are repeated patterns in the samples due to the resampling with replacement.  
• Assumes that the data are representative of the population; this assumption breaks down with highly skewed datasets or when the sample size is too small, leading to biased performance estimates.

**AIC:**  
• Over-penalizes models with many predictors, especially when there is multicollinearity. AIC favors simpler models, which may result in underfitting if it doesn’t capture enough complexity in the data.  
• Assumes that the model is correctly specified. If the underlying model assumptions are incorrect (e.g., nonlinear relationships), AIC may mislead by suggesting an inadequate model.

3. What could you implement given more time to mitigate these cases or help users of your methods?
   Mitigation and Improvements
   For cross-validation:
   Implement stratified k-fold cross-validation for imbalanced datasets.
   For bootstrapping:
   Introduce subsampling without replacement to reduce overfitting risks.
   Combine bootstrapping with ensemble techniques (e.g., bagging) for stability.
   For AIC:
   Complement with other criteria (e.g., BIC) or validate results using predictive metrics.
   Test for multicollinearity and address it before applying AIC.

## **Mitigation and Improvements**

**For cross-validation:**  
• Implement stratified k-fold cross-validation for imbalanced datasets.

**For bootstrapping:**  
• Introduce subsampling without replacement to reduce overfitting risks.  
• Combine bootstrapping with ensemble techniques (e.g., bagging) for stability.

**For AIC:**  
• Complement with other criteria (e.g., BIC) or validate results using predictive metrics.  
• Test for multicollinearity and address it before applying AIC.

## **Parameters Exposed to Users for Model Selectors**

**For Cross-validation:**  
• k: Number of folds used for cross-validation (default is 5). This allows users to specify how many parts the dataset should be split into for validation.

**For Bootstrapping:**  
• n_iterations: Number of bootstrap samples to generate (default is 100). Users can control how many resampling iterations are performed to assess model variability.

**For Regression Models:**  
• Performance evaluation metrics exposed:

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

**For Classification Models:**  
• Performance evaluation metrics exposed:

- Accuracy
- Precision
- Recall
- F1-Score

These parameters allow users to fine-tune their models and select the appropriate evaluation method for their task, whether for regression or classification.
