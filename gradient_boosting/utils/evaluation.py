import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train a model and evaluate its performance on test data.

    """
    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict the target values for the test data
    predictions = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((y_test - predictions) ** 2)

    # Calculate the R-squared score (R²)
    # R² measures the proportion of variance explained by the model
    r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    # Return the computed evaluation metrics
    return mse, r2
