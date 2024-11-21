This project showcases custom implementations of model selection techniques applied to a wine quality dataset. It includes methods such as k-fold cross-validation, bootstrapping, and AIC to evaluate and compare different linear regression models. The goal is to demonstrate the hands-on implementation of machine learning algorithms and evaluation metrics for both regression and classification tasks, all while avoiding pre-built libraries like scikit-learn. This project emphasizes a deeper understanding of the underlying principles of model selection and evaluation.

Features:
Linear Regression from scratch.
k-Fold Cross-Validation and Bootstrapping for performance validation.
Regression and classification metric calculations.
Applicability to different datasets.

Implementation:

1. CustomLinearRegression
   This is a custom implementation of linear regression using the normal equation.
   Attributes:
   • self.coef*: Stores the coefficients (weights) of the model.
   • self.intercept*: Stores the intercept term.
   Methods:
   • fit(X, y):
   o Adds a bias term (intercept) to the input features X by concatenating a column of ones.
   o Uses the normal equation formula:
   𝜃 =(𝑋𝑇𝑋)−1𝑋𝑇𝑦
   where θ is the vector of coefficients and intercept.
   o Extracts the intercept (first element) and stores it separately.
   • predict(X):
   o Calculates predictions as:
   𝑦^ = 𝑋 ⋅coef + intercept\_

2. k fold cross validation
   This function evaluates a model's performance by splitting the dataset into k folds and iteratively training and testing on these folds.
   • Parameters:
   o X, y: Input features and labels.
   o model: The model to train and evaluate.
   o k: Number of folds (default is 5).
   • Steps:
   o Shuffle the dataset indices.
   o Divide the dataset into k approximately equal parts (folds).
   o For each fold:
    Use it as the validation set.
    Use the remaining data as the training set.
    Train the model on the training set.
    Predict and compute Mean Squared Error (MSE) on the validation set.
   o Return the mean and standard deviation of the MSE across all folds.

3. Bootstrapping for Model Selection
   This function evaluates a model's performance by creating multiple resampled datasets.

• Parameters:
o X, y: Input features and labels.
o model: The model to train and evaluate.
o n_iterations: Number of bootstrap iterations.
• Steps:
o For each iteration:
 Create a resampled dataset by sampling with replacement.
 Train the model on the resampled data.
 Evaluate MSE on the original dataset.
o Return the mean and standard deviation of the MSE across iterations.

4. Custom k-Fold Random Forest Implementation
1. Inputs:
   • X: Features matrix (2D array or DataFrame).
   • y: Target variable (1D array or Series).
   • model: A Random Forest model instance (could be a custom implementation or one from libraries like scikit-learn). Here we used out custom model
   • k: Number of folds (default is 5).
1. Steps:
   • Shuffle and Split the Data:
   o Shuffle the dataset indices to ensure random distribution of data.
   o Divide the dataset into k equal parts (folds).
   • Iterative Training and Testing:
   o For each fold:
    Use it as the test set.
    Combine the remaining folds as the training set.
    Train the Random Forest model on the training set.
    Use the trained model to predict on the test set.
    Compute performance metrics (e.g., accuracy, MSE) for the fold.
   • Aggregate Metrics:
   o Store the computed metrics for each fold.
   o Calculate the mean and standard deviation of these metrics across folds.
   • Outputs:
   o Returns the mean and standard deviation of the evaluation metrics.

Algorithm:

1. Initialization
   • The class is initialized with parameters like n_trees, max_depth, and min_samples_split.
2. Training (fit Method)
   • For each tree:
3. Create a bootstrap sample of the dataset.
4. Train a decision tree on this bootstrap sample, selecting a random subset of features at each split.
5. Store the trained decision tree in the trees attribute.
6. Prediction (predict Method)
   • For a given input X:
7. Pass the data through all trained decision trees to obtain individual predictions.
8. Aggregate these predictions:
    For classification: Use majority voting.
    For regression: Compute the average.

In this project, we implemented generic k-fold cross-validation and bootstrapping methods for model selection, showcasing their use in both regression and classification tasks. For regression, we developed a custom k-fold cross-validation approach, a custom linear regression model, and evaluation metrics to assess regression performance, applying these techniques to the Wine Dataset to predict wine quality.
For classification, we implemented a custom k-fold cross-validation method, a custom Random Forest classifier, and evaluation metrics to evaluate classification performance, again using the Wine Dataset.

Additionally, to validate our implementations further, we tested the custom regression model on the Boston Housing dataset and the custom classification model on the Iris dataset, demonstrating their effectiveness and robustness.

1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
   Agreement:
   • In simple cases like linear regression, the custom cross-validation and bootstrapping selectors often align with AIC, provided the dataset fits the assumptions of linear regression (e.g., linearity, homoscedasticity).
   • All three methods evaluate the model's goodness of fit, but they focus on different aspects:
   o AIC evaluates the tradeoff between model complexity (number of parameters) and goodness of fit, favoring models that balance simplicity with fit.
   o Cross-validation focuses on how well the model generalizes to unseen data, ensuring the model performs well on data not seen during training.
   o Bootstrapping assesses how stable the model’s performance is by resampling the data multiple times and measuring variability in performance.
   Disagreement:
   • The methods might disagree in cases of:
   o Nonlinear relationships that violate linear regression assumptions. AIC might still penalize the complexity of the model, while cross-validation and bootstrapping might reflect better or worse performance depending on the model's ability to generalize or overfit.
   o Small datasets where both cross-validation and bootstrapping can yield unstable results due to high variance in performance estimates.
   o High collinearity among predictors in linear regression, where AIC might favor simpler models that avoid overfitting, but cross-validation and bootstrapping may still indicate overfitting or instability with more predictors.
2. In what cases might the methods you've written fail or give incorrect or undesirable results?
   Cross-validation:
   • Computationally expensive for large datasets: Since each fold requires retraining the model, this can be prohibitively slow for large datasets.
   • Biased results on imbalanced datasets or non-i.i.d. data (e.g., time series), where standard k-fold cross-validation might overestimate model performance due to improper data partitioning.
   Bootstrapping:
   • May overfit the model to specific patterns in the data, especially when there are repeated patterns in the samples due to the resampling with replacement.
   • Assumes that the data are representative of the population; this assumption breaks down with highly skewed datasets or when the sample size is too small, leading to biased performance estimates.
   AIC:
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
4. What parameters have you exposed to your users in order to use your model selectors?
   • For Cross-validation:
   o k: Number of folds used for cross-validation (default is 5). This allows users to specify how many parts the dataset should be split into for validation.
   • For Bootstrapping:
   o n_iterations: Number of bootstrap samples to generate (default is 100). Users can control how many resampling iterations are performed to assess model variability.
   • For Regression Models:
   o Performance evaluation metrics exposed:
    MSE (Mean Squared Error)
    RMSE (Root Mean Squared Error)
    MAE (Mean Absolute Error)
    R² (Coefficient of Determination)
   • For Classification Models:
   o Performance evaluation metrics exposed:
    Accuracy
    Precision
    Recall
    F1-Score
   These parameters allow users to fine-tune their models and select the appropriate evaluation method for their task, whether for regression or classification.

Links for Additional Datasets used for testing:

Iris: https://www.kaggle.com/datasets/himanshunakrani/iris-dataset
Boston Housing: https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd
