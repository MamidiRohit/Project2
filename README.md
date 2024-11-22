# Project 2
# Model Selection
Nidhi Shrivastav A20594009 nshrivastav@hawk.iit.edu 
Rutuja Jadhav A20539073 rjadhav4@hawk.iit.edu 
Pankaj Jagtap A20543260 pjagtap1@hawk.iit.edu

A. Overview: This implementation evaluates a linear regression model using k-fold cross-validation and bootstrapping. In k-fold cross-validation, the dataset is split into 'k' parts, with the model trained on different subsets and tested on the remaining one, providing insights into its performance across multiple data splits. Bootstrapping involves generating random samples, training the model on each, and evaluating its performance to assess variability and generalization.The model is evaluated using metrics like MSE, MAE, and R² to measure accuracy, error, and fit quality. Users control parameters such as the number of samples, k-fold folds, shuffle options, training size for bootstrapping, and choice between k-fold and bootstrapping.

B. Files 
• lib.py: Contains the core implementation of the ModelSelector class and utility functions for model training, evaluation, and metric computation. 
• test.py: Includes unit tests for validating the functionality of the ModelSelector class and its components using pytest.
• main.py: Provides an example workflow demonstrating how to use the ModelSelector framework with a sample dataset for model selection and performance evaluation.

C. Implementation Details 
Library Module (lib.py):
1. LinearRegression Class: Implements linear regression using the normal equation.
Includes:
fit: Trains the model with bias and regularization to prevent singularity.
predict: Makes predictions using the fitted weights.
2. Metrics Functions:
mean_squared_error: Computes the Mean Squared Error (MSE).
mean_absolute_error: Computes the Mean Absolute Error (MAE).
r2_score: Calculates R² with adjustments to prevent division by zero.
3. Validation Techniques:
k_fold_cross_validation: Evaluates the model with k-fold splits, computes average MSE, MAE, and R².
bootstrapping: Performs bootstrapping to validate the model over several epochs.
4. Synthetic Data Generation:
generate_data: Creates test data with random features and weights.

Main Execution (main.py):
Features:
Allows user-defined data size (n_samples, n_features).
Two validation methods:
k-Fold Cross-Validation:
Takes input for the number of folds (k) and shuffle preference.
Bootstrapping:
Accepts the size of training data (s) and number of epochs.
Outputs average metrics (MSE, MAE, R²).
Interactivity:
Validates user input.
Prints results for chosen validation methods.

Testing Suite (test.py) - Detailed Breakdown:
Tested Components:
1. LinearRegression Class:
Fit Method: Validates that the weights are correctly computed after training using sample data.
Predict Method: Ensures the model generates predictions matching the input data shape.
Prediction Accuracy: Tests whether the predictions closely align with the expected outcomes (ground truth), using a tolerance of one decimal place for numerical stability.
2. Metrics Functions:
Mean Squared Error (MSE): Validates the computed error matches the expected value for given synthetic inputs, ensuring precise error calculation.
Mean Absolute Error (MAE): Confirms that the absolute errors are averaged correctly and checks against predefined tolerances for floating-point operations.
R² Score: Ensures the coefficient of determination correctly evaluates model performance. It checks edge cases like near-zero variance with a small epsilon to avoid division by zero.
3. Validation Techniques:
k-Fold Cross-Validation: Verifies that the metrics (MSE, MAE, and R²) are computed correctly for each fold and that the averages across folds are accurate. Ensures shuffled data handling is implemented properly.
Bootstrapping: Tests the method's ability to sample with replacement, calculate metrics for out-of-sample data, and aggregate the results across multiple epochs.
4. Data Generator:
Confirms that the generated feature matrix (X) and target vector (y) match the specified dimensions.
Verifies that the output includes noise and random variations consistent with the requirements for synthetic testing.

Framework:
Pytest Usage:
Leverages pytest for structured and modular testing.
Fixtures (@pytest.fixture) are used to set up reusable sample data for LinearRegression and metrics validation, enhancing test readability and reducing redundancy.

Assertions:
Prediction Validations:
Ensures predicted values align with expected outputs for sample data using numpy’s assert_almost_equal for numerical comparison.

Metrics Accuracy:
Compares computed metrics against manually calculated expected values, with tolerances to account for floating-point variations.
Output Structure:
Confirms that returned objects (e.g., metrics dictionaries and averages) have the correct keys and expected lengths.
Error Handling:
Incorporates tolerances and edge case validations (e.g., empty out-of-sample data in bootstrapping) to ensure robustness.

To execute the code, follow these steps:
1. Set up the environment: Install required dependencies.
2. Prepare the data:Generate synthetic data or load a dataset. Define parameters like the number of samples and features. Then, select the model selection method (kFold or Bootstrapping) and pass the necessary parameters such as kFold (number of folds), shuffle option, bootstrap training size, and epochs.
3. Check results: The script will output evaluation metrics like MSE, MAE, and R².
4. Run tests: To validate functionality, run pytest test.py to execute tests for predictions and metrics.

Questions
1. Do cross validation and bootstrapping agree with AIC in simple cases?  
Based on our implementation of the linear regression model, cross-validation and bootstrapping often agree with AIC in simple cases. AIC evaluates model performance by balancing the fit to the data with model simplicity, while my methods use metrics like MSE, MAE, and R² to measure accuracy and fit. Since all these methods focus on assessing model performance, their conclusions tend to align in straightforward cases like linear regression, especially when MSE is used to evaluate model error.

2. When might your methods fail or give bad results?  
Our methods might fail or give bad results in cases such as:
Data Issues: Small, unrepresentative datasets or those with outliers and noise can lead to unreliable metrics like MSE, MAE, and R².
Model Assumption Violations: Linear regression assumes linearity, independence of errors, and homoscedasticity. Violations of these assumptions can distort performance evaluations.
Overfitting/Underfitting: Cross-validation and bootstrapping may not effectively capture these issues if the splits don't represent the data distribution well.
High Dimensionality or Multicollinearity: A large number of features or highly correlated features can destabilize coefficient estimates, impacting metrics.
Improper Preprocessing: Lack of data scaling or standardization can bias metrics, as MSE and MAE are sensitive to scale differences.
Metric Limitations: R² may give misleading results on datasets with low variance and doesn't directly penalize overfitting. 

3. What could you improve with more time?  
Stratified k-fold: Ensuring balanced splits by stratifying the data based on target variables would help maintain consistency, especially in imbalanced datasets.
Bootstrapping sample control: For small datasets, it is important to ensure that bootstrapping samples do not lead to overfitting or poorly generalized models, possibly by limiting sample size or using alternative resampling techniques.
Automating parameter choices: Automating the selection of hyperparameters like k (for cross-validation) and training set size would optimize model performance and make the process more efficient.

4. What parameters do users control?  
Data: Number of samples and features (data provided for training and testing).
kFold: Number of folds (k) and shuffle option (whether to randomize the data before splitting).
Bootstrapping: Sample size (number of samples to draw) and training size (number of samples used for training in each bootstrap iteration).
Model: Choice between k-fold cross-validation and bootstrapping for model evaluation. 
