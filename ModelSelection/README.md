# Project 2

# Group Name: A20560777

# Group Members:
Pooja Pranavi Nalamothu (CWID: A20560777)

# Model Selection

1.	Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?

Yes. For basic problems like linear regression, cross validation and boot strap techniques are found to select the models which are equally selected by AIC in most of the cases. They both try to assess the model’s suitability for classifying unknown data. Holds its measures on prediction errors of cross-validation while AIC assesses the goodness of the fit and charges the model complexity. According to Li et al., when the assumptions of AIC (like errors’ normally distributed nature) are valid, the results match perfectly. However, cross validation can be more robust when assumptions that lead to it are not met.

2.	In what cases might the methods you've written fail or give incorrect or undesirable results?

Cross-Validation:

•	Small Datasets: Finally, while k-fold cross-validation may produce high variance when used in datasets with little samples because of limited data within each fold.

•	Imbalanced Data: If the dataset is imbalanced, these particular classes can be minority samples in some folds which can cause a skewed assessment.

•	Overfitting in Small Folds: If the model is too flexible, it may fit the training subsets which are small in size.

Bootstrapping:

•	Overlap in Training and Test Data: It should be noted that in bootstrap samples the observations can be repeated in both sets: training and test, which causes the over estimation of the performance.

•	Highly Complex Models: The major draw back of bootstrapping is that it may not produce the right estimate of prediction error when working with models that are very sensitive to small changes in data.

•	High Bias in Small Datasets: Bootstrap techniques may give a biased estimation if the size of the current dataset is virtually small; the amount of unseen samples during each bootstrap repetition reduces.

3.	What could you implement given more time to mitigate these cases or help users of your methods?

•	Stratified Cross-Validation: For unrealistic data split, applying the technique of stratified k-fold cross validation where each fold in the dataset shall have reasonable distribution of classes.

•	Leave-One-Out Cross-Validation (LOOCV): However, LOOCV should only be applied on very small datasets so as to minimize variance when evaluating the model.

•	Improved Bootstrapping Techniques: Perform the use the of .632+ bootstrap estimator so as to reduce the over estimation bias that arises when using bootstrap predict from over fitted models.

•	Time-Series Specific Validation: Add rolling cross-validation or other similar to it that is better suited for time series or dependent data.

•	Regularization Support: Implementation of automatic hyperparameter tuning of regularized models such as ridge regression and Lasso regression to reduce overfitting.

•	Model Diagnostics: Integrated diagnostic plots or metrics will help to detect overtraining, underfitting or unbalanced data problem.

•	Parallelization: For faster execution on large datasets, replicate k-fold or bootstrap computations where the experiment is the different subset of data.

4.	What parameters have you exposed to your users in order to use your model selectors?

For Cross-Validation:

•	k: K parameter of k-fold cross validation, it represents number of folds which divides data set into two sets- training and testing.

•	random_state: A seed for reproducibility when shuffling the data comes in handy.

•	loss_function: Another function created by the user which computes the loss, these can be MSE, MAE and other such pertinent parameters.

For Bootstrapping:

•	B: Number of bootstrap samples.

•	loss_function: A function defined by a user that is meant to capture the loss (for example a measure of variance, accuracy).
•	alpha: A parameter when constructing the model in the form of a decision tree, allows for calculating confidence intervals of the indicator of model quality.

For Models:

•	Learning rate and the number of iterations for an algorithm as well as some parameters of the learning problem like normalization.

How to run the code

Step 1:

Clone or download the code: Ensure all the datasets are in the correct paths as mentioned in the code.
Step 2:

Install necessary dependencies
 
Step 3:

Run the script
 
Step 4:

Ensure all the datasets are properly loaded and called

Note: Update paths in the script, if required.

Basic Usage:

•	For Synthetic Data, you can run this code directly

CODE:

```python
import numpy as np

class ModelSelection:

    def __init__(self, model, loss_function):
        """
        Initialize the model selector with a given model and loss function.

        Parameters:
        - model: A class with `fit` and `predict` methods.
        - loss_function: A callable that takes (y_true, y_pred) and returns a scalar loss.
        """
        self.model = model
        self.loss_function = loss_function

    def k_fold_cross_validation(self, X, y, k=5):
        """
        Perform k-fold cross-validation.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - k: Number of folds (default is 5).

        Returns:
        - mean_loss: The average loss across all folds.
        """
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k
        losses = []

        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            loss = self.loss_function(y_test, y_pred)
            losses.append(loss)

        mean_loss = np.mean(losses)
        return mean_loss

    def bootstrap(self, X, y, B=100):
        """
        Perform bootstrap resampling to estimate prediction error.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - B: Number of bootstrap samples (default is 100).

        Returns:
        - mean_loss: The average loss across all bootstrap samples.
        """
        n = len(y)
        losses = []

        for _ in range(B):
            bootstrap_indices = np.random.choice(np.arange(n), size=n, replace=True)
            oob_indices = np.setdiff1d(np.arange(n), bootstrap_indices)

            if len(oob_indices) == 0:  # Skip iteration if no OOB samples
                continue

            X_train, X_test = X[bootstrap_indices], X[oob_indices]
            y_train, y_test = y[bootstrap_indices], y[oob_indices]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            loss = self.loss_function(y_test, y_pred)
            losses.append(loss)

        mean_loss = np.mean(losses)
        return mean_loss

    def evaluate_model(self, X, y, method='k_fold', **kwargs):
        """
        Evaluate the model using the specified method.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - method: 'k_fold' or 'bootstrap'.
        - kwargs: Additional parameters for the evaluation method.

        Returns:
        - loss: The evaluation loss.
        """
        if method == 'k_fold':
            return self.k_fold_cross_validation(X, y, **kwargs)
        elif method == 'bootstrap':
            return self.bootstrap(X, y, **kwargs)
        else:
            raise ValueError("Unsupported method. Choose 'k_fold' or 'bootstrap'.")
```

#Example of a simple linear regression model
```python
class SimpleLinearModel:
    def fit(self, X, y):
        self.coef_ = np.linalg.pinv(X) @ y

    def predict(self, X):
        return X @ self.coef_

#Mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#Create synthetic data
np.random.seed(42)
X = np.random.rand(100, 3)
y = X @ np.array([1.5, -2.0, 1.0]) + np.random.randn(100) * 0.1

#Initialize model and model selector
model = SimpleLinearModel()
selector = ModelSelection(model, mean_squared_error)

#Perform k-fold cross-validation
k_fold_loss = selector.evaluate_model(X, y, method='k_fold', k=5)
print("K-Fold Cross-Validation Loss:", k_fold_loss)

#Perform bootstrap
bootstrap_loss = selector.evaluate_model(X, y, method='bootstrap', B=100)
print("Bootstrap Loss:", bootstrap_loss)

model.fit(X, y)

#Evaluate
predictions = model.predict(X)
mse = np.mean((y - predictions) ** 2)
print(f"Mean Squared Error: {mse:.4f}")
print("First 10 Predictions:", predictions[:10])
```

•	For other datasets

Ensure the correct dataset is loaded and is in the correct directory.

Define X and y for the corresponding features and target variables.

Once you've modified the file path, you can use pytest to run all tests, including real-world datasets and synthetic dataset

•	Open a terminal in the directory containing your test script.

•	Run Pytest with the following command: pytest test_datasetname.py

 
