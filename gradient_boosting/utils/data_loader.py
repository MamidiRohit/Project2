import pandas as pd

def load_data(url, target_column, train_ratio=0.8):
    """
    Load data from a given URL, split into training and testing sets.

    Parameters:
    - url: The URL to the CSV file containing the dataset.
    - target_column: The name of the target column in the dataset.
    - train_ratio: The ratio of the data to be used for training (default is 0.8).

    Returns: X_train,X_test,y_train,y_test
    """
    # Read the CSV file from the provided URL
    data = pd.read_csv(url)

    # Separate features (X) and target variable (y)
    X = data.drop(columns=[target_column]).values  # Drop the target column to get features
    y = data[target_column].values  # Extract the target column

    # Determine the number of training samples based on the train_ratio
    n_train = int(len(X) * train_ratio)

    # Split the data into training and testing sets
    X_train = X[:n_train]  # Features for training
    X_test = X[n_train:]  # Features for testing
    y_train = y[:n_train]  # Target values for training
    y_test = y[n_train:]  # Target values for testing

    # Return the training and testing splits
    return X_train, X_test, y_train, y_test
