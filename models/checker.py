"""
Utility functions for data preprocessing and transformation.

Functions:
-----------
1. fill_if_null(data):
    Fills null values in the dataset with the mean of their respective columns.

2. check_null(data):
    Checks for null values and fills them using `fill_if_null` if any are found.

3. XandY(data, dept):
    Splits the dataset into feature matrix (X) and target vector (Y) based on the target column name.

4. normalize_data(X):
    Normalizes the feature matrix to scale values between 0 and 1 for each column.

Each function ensures robust handling of data for preprocessing tasks in machine learning workflows.
"""

import numpy as np

def fill_if_null(data):
    null_boy = np.array(data.columns[data.isnull().any()])
    for i in null_boy:
        data[i] = data[i].fillna(data[i].mean())
    return data


def check_null(data):
    print("\n\nChecking for the null value present in the data")
    if data.isnull().values.any():
        fill_if_null(data)
        print(data.isnull().sum())
    else:
        print(data.isnull().sum())


def XandY(data, dept):

    Y = data[dept].to_numpy()
    data.drop(dept, axis=1, inplace=True)
    X = data.to_numpy()

    return [X, Y]


def normalize_data(X):

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    X_normalized = (X - min_vals) / range_vals
    return X_normalized, min_vals, max_vals




