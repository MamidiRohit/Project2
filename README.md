> ZIRUI OU A20516756

## Model Selection

### Program Design
![alt text](images\image.png)

### Results Overview
| Metric                | **Linear Regression**       | **Ridge Regression**       |
|-----------------------|-----------------------------|-----------------------------|
| Coefficients          | [0.2151, 0.5402]           | [0.2411, 0.4849]           |
| K-Fold MSE            | 0.8418                     | 0.8471                     |
| Bootstrapping MSE     | 0.8219                     | 0.8214                     |
| AIC                   | -17.495                    | -17.461                    |
| SSR                   | 80.6585                    | 80.6853                    |
| Verification (Coefficients) | True                     | True                       |
| Verification (AIC)    | True                       | True                       |
![alt text](images\Figure_1.png)

### How to Run the Code

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jerry-zirui/Project2.git
   cd Project2
   ```

2. **Install Required Packages:**
   ```bash
   pip install numpy
   pip install matplotlib #optional
   ```

3. **Generate Synthetic Data**
    ```python
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    n = 100  # Number of samples
    p = 1    # Number of features
    X = np.random.rand(n, p)  # Feature matrix
    beta_true = np.array([1])  # True coefficients
    y = X @ beta_true + np.random.randn(n)  # Target variable with noise
    ```

4. **Define Models**
    ```python
    models = {
        "Linear Regression": {"fn": linear_regression},
        "Ridge Regression": {"fn": ridge_regression, "params": {"alpha": 1.0}},
    }
    ```

5. **Evaluate Models**
    ```python
    # Evaluate models
    results = evaluate_models(models, X, y, k=5, B=1000)
    ```

6. **Visualize Results**
    ```python
    # Visualize results
    results = visualize(evaluate_models, X, y, models, results)
    ```

7. **Run the Model Selection Script:**
    ```bash
    python model_selection.py
    ```
### Q&A

#### Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
  - In my case with linear regression and Ridge Regression models, the cross-validation and bootstrapping model selectors generally agree with the AIC model selector.
  - The MSE values from cross-validation and bootstrapping are similar, and the AIC values are consistent with the model selection.

#### In what cases might the methods you've written fail or give incorrect or undesirable results?
- **Multicollinearity:**
  - If there are highly correlated features, the methods may give incorrect results due to multicollinearity. For example, in this dataset, if `Feature1` and `Feature2` were highly correlated, the coefficients might not accurately reflect their individual contributions.
  
- **Small Sample Sizes:**
  - With small sample sizes, the methods may not have enough data to provide reliable estimates. This could lead to high variance in the model coefficients and MSE values.
  
- **High Variance Models:**
  - With high variance models, the methods may give higher MSE values and less desirable results.
  
- **Poorly Specified Models:**
  - If the models are not properly specified (e.g., incorrect regularization parameters), the methods may fail. For instance, if the Ridge Regression model's alpha parameter was set too high, it might over-shrink the coefficients, leading to underfitting.

#### What could you implement given more time to mitigate these cases or help users of your methods?
- **Feature Selection:**
  - Implement feature selection methods to handle multicollinearity. Techniques like Variance Inflation Factor (VIF) can be used to identify and remove highly correlated features.
  
- **Regularization Techniques:**
  - Implement more advanced regularization techniques such as Lasso, Ridge, and Elastic Net to handle different types of data and model requirements.
  
- **Cross-Validation with Multiple Splits:**
  - Use cross-validation with multiple splits to improve reliability. This can help in obtaining a more stable estimate of the model performance.
  
- **Bootstrap Aggregation:**
  - Implement bootstrap aggregation to improve the stability of the results. This technique can help in reducing the variance of the model estimates.
  
- **Model Diagnostics:**
  - Provide detailed diagnostics to help users understand the model selection process. This can include residual plots, coefficient plots, and other visualizations that can aid in interpreting the results.

### What parameters have you exposed to your users in order to use your model selectors.
- **ridge_regression**
  -  `alpha`: which controls the strength of the L2 regularization
- **K-Fold Cross-Validation:**
  - `k`: Number of folds. (Default: 5)
  - `model_fn`: Model function. (e.g., `LinearRegression()`, `Ridge()`)
  - `model_params`: Model parameters. (e.g., `{'alpha': 1.0}` for Ridge Regression)

- **Bootstrapping:**
  - `B`: Number of bootstrap samples. (Default: 1000)
  - `model_fn`: Model function. (e.g., `LinearRegression()`, `Ridge()`)
  - `model_params`: Model parameters. (e.g., `{'alpha': 1.0}` for Ridge Regression)