**Group Members:-**

**Chandan Kumar - CWID - A20525237**

**Nanda Kishore Thummala - CWID - A20595194**

**Niharika Bandaru - CWID - A20600363**

To run the python code you can use this **"python -m generic-kfold-bootstrap.tests.test_model_selector"**

# Questions

**1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?**

In this case, AIC and BIC values are consistent with the relatively low MSE values from cross-validation and bootstrap. Since all metrics (CV MSE, Bootstrap MSE, AIC, and BIC) indicate reasonable fit and consistency without drastic discrepancies, we can say they align in terms of suggesting that this linear regression model is a good fit for the data.

Here’s how to interpret the alignment:

**Bootstrap MSE (1.3516)** is slightly lower than **CV MSE (1.7455)** only with 50 records of data, also we increased the amount of data to 3551 records then we got **k-fold CV MSE: 1.2206262241189405** and
**Bootstrap MSE: 1.2175121427152984**, which is common when the model performs reliably on different samples but sees a slight improvement with additional bootstrap sampling.
AIC and BIC are in a lower range, suggesting this model balances fit and simplicity.
Conclusion: In this simple linear regression case, all selectors (AIC, BIC, CV MSE, and Bootstrap MSE) agree that this model fits the data without indicating overfitting or underfitting.

**2. In what cases might the methods you've written fail or give incorrect or undesirable results?**

Overfitting in complex models: For models with many features or complex structures, cross-validation and bootstrap methods might give low MSE due to overfitting, while AIC and BIC might rise, indicating the model is too complex.

Small sample sizes: With very small datasets, bootstrap samples might lack variance, leading to an underestimated MSE.

Non-normal residuals: AIC and BIC assume that errors are normally distributed, which may not hold for all datasets, leading to potentially misleading values.

High-dimensional data: If the number of features is close to the number of observations, AIC and BIC could be unreliable as they heavily penalize models with many parameters relative to data size.

**3. What could you implement given more time to mitigate these cases or help users of your methods?**

Residual Analysis: Add residual analysis to check assumptions of normality and homoscedasticity (constant variance) for both AIC and BIC validity.

Adjustable Regularization: Incorporate regularization terms (e.g., Lasso or Ridge) to handle overfitting in high-dimensional models.

Error Distributions: Allow users to specify alternative error distributions for AIC/BIC when normality doesn’t hold, improving robustness for a variety of data types.

Cross-validation on a Range of Models: Implement automated model selection across different model types (e.g., linear, polynomial) to help users find the best fit.

**4. What parameters have you exposed to your users in order to use your model selectors?**

k: The number of folds for k-fold cross-validation (default is 5).

n_bootstraps: Number of bootstrap samples (default is 100).

random_state: Random seed for reproducibility.

Model Choice: The choice of model (e.g., linear regression) can be set by the user, allowing flexibility in applying the selector to various models.

Each of these parameters enables users to customize cross-validation and bootstrap settings and choose their model, allowing the methods to adapt to different dataset characteristics and modeling goals.











