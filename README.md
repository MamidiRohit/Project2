# Project 2: MODEL SELECTION

**Course:** CS584 - Machine Learning <br>
**Instructor:** Steve Avsec<br>
**Group Members:**
- ssaurav@hawk.iit.edu (FNU Saurav) - A20536122
- psavant@hawk.iit.edu (Pallavi Savant) - A20540976


## Project Overview

This project implements two model selection techniques:
1. **k-Fold Cross-Validation**:
   - Evaluates a machine learning model by splitting the dataset into \( k \) folds and using each fold as a validation set while training on the remaining \( k-1 \) folds.
   - The average loss (e.g., Mean Squared Error) across all folds is calculated to estimate the model’s predictive performance.

2. **Bootstrapping**:
   - Evaluates a model by generating multiple bootstrap samples (random sampling with replacement) from the dataset and using the out-of-bag (OOB) samples for validation.
   - The average error across all bootstrap iterations is computed to measure model performance.

Both methods are implemented for general-purpose models that provide `fit()` and `predict()` methods.

## Code Files
1. **`main.py`**:
   - Demonstrates the usage of the implemented k-fold cross-validation and bootstrapping methods.
   - Uses a simple linear regression model on synthetic data as an example.

2. **`linear_regression.py`**:
   - Implements a basic Linear Regression model using the **Normal Equation**.
   - Includes methods for fitting the model (`fit()`) and making predictions (`predict()`).

3. **`model_selection.py`**:
   - Contains implementations for:
     - **k-Fold Cross-Validation**: Evaluates model performance using \( k \)-fold splitting.
     - **Bootstrapping**: Evaluates model performance using random sampling with replacement.

## Functions Overview

| **Function**                | **Description**                                                                                       |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| **`k_fold_cross_validation`** | Performs k-fold cross-validation on the given model by splitting the data into `k` folds, training on `k-1` folds, and testing on the remaining fold. Returns the average error across all folds. |
| **`bootstrapping`**          | Implements bootstrapping with out-of-bag (OOB) evaluation by resampling data with replacement. Calculates the average error across all bootstrap iterations. |
| **`LinearRegression.fit`**   | Fits a linear regression model to the given training data using the Normal Equation. Calculates weights and intercept for the model. |
| **`LinearRegression.predict`** | Predicts target values for the given input data based on the weights and intercept obtained from the `fit` method. |
| **`main.py`**                | Combines all components to generate synthetic data, perform k-fold cross-validation and bootstrapping, and print the results. |

## How to Run the Code

1. Clone the repository and navigate to the project directory.
2. Install the required Python libraries:
   ```bash
   pip install numpy scikit-learn
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
4. Observe the cross-validation and bootstrapping errors printed in the console.
    ```yaml
    5-Fold Cross-Validation Error (MSE): 0.9938904780907099
    Bootstrap Error (MSE): 1.0599287186388127
    ```

## Key Questions

### 1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?

Yes, in simple cases like linear regression, cross-validation and bootstrapping often agree with a simpler model selector like the Akaike Information Criterion (AIC).
- **Cross-validation and bootstrapping** directly estimate the model's predictive performance on unseen data by simulating multiple train-test splits.
- **AIC (Akaike Information Criterion)** on the other hand, has been built in to prevent over-fitting in model and hence the more complex ones are discouraged although, there exists an underlying likelihood function assumed.

---

### 2. In what cases might the methods you've written fail or give incorrect or undesirable results?

The methods may fail or give incorrect results in the following cases:
1. **Imbalanced Datasets**:
   - If the dataset is highly imbalanced, random splits in cross-validation or bootstrapping may fail to represent minority classes adequately, leading to biased error estimates.
2. **Small Datasets**:
   - With limited data, random splitting in both methods might cause high variance in error estimates due to insufficient training data or small validation sets.
3. **Correlated features or data points**:
   - Cross-validation and bootstrapping may underestimate errors in models trained on data with correlations between features if the validation set doesn't reflect the correlation structure.
4. **Violated model assumptions**:
   - We have used linear regression and handled the edge cases and assumptions. However, if any other model is used, it might cause issues as it may rely on the model's assumptions being appropriate for the dataset.
   - If the model's assumptions are violated, the methods may yield unreliable performance estimates.
5. **Computational Constraints**:
   - For very large datasets, the computational overhead of these methods might be impractical without optimized implementations.
6. **Non-IID data**:
   - If data isn't independent and identically distributed, random splits may ignore dependencies, resulting in unreliable error estimates.
7. **Overlap in bootstrapping samples**:
   - Bootstrapping involves sampling with replacement, which means some samples can appear multiple times in a single bootstrap iteration.
   - This can bias the model towards overfitting specific samples in the training set, particularly in small datasets.

---

### 3. What could you implement given more time to mitigate these cases or help users of your methods?

1. **Balanced k-Fold Cross-Validation**:
   - For imbalanced datasets, we can implement balanced sampling to ensure proportional representation of each class in all folds.
   - This will provide unbiased error estimates for minority classes.
2. **Parallelization**:
   - Can use parallel computing libraries to speed up the computation for both cross-validation and bootstrapping.
   - This can make the methods practical for large datasets or complex models.
3. **Dimensionality Reduction**:
   - We can Address challenges with high-dimensional datasets and add automatic feature selection or dimensionality reduction techniques to handle datasets with correlated features or high dimensionality.
4. **Handling Overlap in Bootstrapping**:
   - Add checks to identify excessive sample repetition in bootstrapping iterations and adjust sampling strategies dynamically to ensure diversity in bootstrap samples.
5. **Blocked Cross-Validation**:
   - For non-IID data, can implement blocked or grouped cross-validation. This ensures that dependencies are maintained by splitting the data into meaningful groups.
6. **OOB Evaluation Improvements**:
   - More robust handling of bootstrap edge cases.
7. **Documentation and Examples**:
   - Provide more detailed documentation. Helps users understand and apply the methods effectively.
6. **Error handling**:
   - Add more error handling for scenarios where methods might for example in cases like too few data points, highly imbalanced data or strong correlations in features.

---

### 4. What parameters have you exposed to your users in order to use your model selectors?

#### **k-Fold Cross-Validation Parameters**:
1. **`model`**: A machine learning model with `fit()` and `predict()` methods.
2. **`X` and `y`**: Input features and target values.
3. **`k`**: Number of folds (default = 5).
4. **`random_seed`**: For reproducibility of random splits. (optional).

#### **Bootstrapping Parameters**:
1. **`model`**: A machine learning model with `fit()` and `predict()` methods.
2. **`X` and `y`**: Input features and target values.
3. **`n_iterations`**: Number of bootstrap iterations (default = 100).
4. **`test_size`**: Proportion of OOB samples (default = 0.3).
5. **`random_seed`**: Seed for reproducibility (optional).
