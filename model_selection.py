# model_selection.py

import numpy as np
from linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error  # For scoring

def k_fold_cross_validation(model, X, y, k=5, random_seed=None):
    """
    Performs k-fold cross-validation on the given model.

    Parameters:
        model (object): The machine learning model to train and evaluate.
        X (numpy.ndarray): The input features for training, with shape (n_samples, n_features).
        y (numpy.ndarray): The target values, with shape (n_samples,).
        k (int): The number of folds (default is 5).
        random_seed (int, optional): A random seed for reproducibility.

    Returns:
        float: The average mean squared error (MSE) across all k folds.
    """

    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    errors = []

    for fold in range(k):
        # Split data into training and validation sets
        val_indices = indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices = np.delete(indices, val_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        #fit the custom linear regression model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Calculate and store the error (MSE)
        error = mean_squared_error(y_val, y_pred)
        errors.append(error)

    # Average the errors
    avg_error = np.mean(errors)
    return avg_error

def bootstrapping(model, X, y, n_iterations, test_size=0.3, random_seed=None):
    """
    Performs bootstrapping with out-of-bag (OOB) evaluation on the given model.

    Parameters:
        model (object): The machine learning model to train and evaluate.
        X (numpy.ndarray): The input features for training, with shape (n_samples, n_features).
        y (numpy.ndarray): The target values, with shape (n_samples,).
        n_iterations (int): The number of bootstrap samples to generate.
        test_size (float): The proportion of the dataset to use as OOB samples (e.g., 0.3 for 30%).
        random_seed (int, optional): A random seed for reproducibility.

    Returns:
        float: The average mean squared error (MSE) for OOB samples across all iterations.
        None: If there are no OOB samples (e.g., if test_size is too small).
    """

    np.random.seed(random_seed)
    n_samples = len(X)
    errors = []

    for _ in range(n_iterations):
        # Bootstrap resample
        train_size = int(n_samples * (1 - test_size))
        resample_indices = np.random.choice(n_samples, size=train_size, replace=True)
        X_resampled, y_resampled = X[resample_indices], y[resample_indices]
        
        # Determine out-of-bag samples based on test_size
        oob_mask = ~np.isin(np.arange(n_samples), resample_indices)
        X_oob, y_oob = X[oob_mask], y[oob_mask]

        # Only evaluate if there are OOB samples
        if len(y_oob) > 0:
            #fit the custom linear regression model
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_oob)

            # Calculate and store the error (MSE)
            error = mean_squared_error(y_oob, y_pred)
            errors.append(error)

    # Average the errors
    avg_error = np.mean(errors) if errors else None
    return avg_error
