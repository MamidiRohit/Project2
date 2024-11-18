"""
References:
    1. AIC definition: Wikipedia - Akaike information criterion
      [LINK] https://en.wikipedia.org/wiki/Akaike_information_criterion
    2. Log likelihood formula - StatLect "Log-likelihood"
      [LINK] https://www.statlect.com/glossary/log-likelihood
"""
import json

import numpy as np
from sklearn.linear_model import *

# Collection of auxiliary functions

"""
    Parameter setting read
"""


def get_param(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


"""
    Model selections
"""


def get_model(model_type: str):
    """
    get_model()
    This function returns the model object for the given model_type defined in 'params,py'.
    If no model_type is defined, it returns default model object as LinearRegression.
    :param model_type: the type of model to use [options] "LinearRegression(default)", "LogisticRegression"
    :return: a model object to train.
    """
    print(f"Model Type: {model_type}")

    if model_type == "LinearRegression":
        return LinearRegression()
    elif model_type == "LogisticRegression":
        return LogisticRegression(max_iter=1000)
    else:  # Invalid type described
        print(f"[Warning] Invalid model type has given. Uses default model type as 'LinearRegression'")
        return LinearRegression()


def get_metric(metric_type: str):
    """
    get_metric()
    This function returns the metric function for the given metric_type defined in 'param.txt'.
    If no metric_type is defined, it returns default setting as MSE.
    :param metric_type: the type of metric to use [options] "MSE(default)", "Accuracy score"
    :return: a metric function to apply.
    """
    print(f"Metric Type: {metric_type}")

    if metric_type == "MSE":
        return MSE
    elif metric_type == "Accuracy score":
        return accuracy_score
    elif metric_type == "R2":
        return R2
    else:  # Invalid type described
        print(f"[Warning] Invalid metric type has given. Uses default metric type as 'MSE'")
        return MSE


"""
    Data import/create functions
"""


def read_csv(file_name: str, test_ratio: float) -> tuple:
    """
    read_csv()
    This function reads the data from the given csv file
    * This function accepts the following valid file format:
        a dataset with either a single or multiple features and a single label,
        where all values excluding the header are numeric.
    :param file_name: name of the file to read
    :param test_ratio: test ratio from the created dataset
    :return:
        X: Dataset with features
        y: Dataset with labels
        train_X, train_y, test_X, test_y: Split dataset by split_dataset() function
    """
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)  # Skip header

    X = data[:, :-1]
    y = data[:, -1]

    # Split dataset into train and test
    train_X, train_y, test_X, test_y = split_dataset(X, y, test_ratio)
    return X, y, train_X, train_y, test_X, test_y


def generate_data(size: int, dimension: int, correlation: float, noise_std: float, random_state: int,
                  test_ratio: float) -> tuple:
    """
    generate_multi_collinear_data()
    This is a function that generates data with multi_collinearity
    The data produced from this function is intended to measure the ElasticNet model's ability
    to handle multi_collinearity, which is a critical feature of the model

    :param size: Number of samples to generate
    :param dimension: Number of features to generate
    :param correlation: Correlation coefficient between features (1 is the worst)
    :param noise_std: Set noise scale to test durability of the trained model
    :param random_state: random seed
    :param test_ratio: test ratio from the created dataset
    :returns:
        X: Dataset with features
        y: Dataset with labels
        train_X, train_y, test_X, test_y: Split dataset by split_dataset() function
    """
    if random_state:
        np.random.seed(random_state)

    # random base sample (1st feature)
    X_base = np.random.rand(size, 1)

    # Create another feature based on the data for the first feature
    X = X_base + correlation * np.random.randn(size, dimension) * noise_std

    # Increase the correlation between each feature (e.g. through linear combination)
    for i in range(1, dimension):
        X[:, i] = correlation * X_base[:, 0] + (1 - correlation) * np.random.randn(size)

    # create weights, bias, and noise
    weights = np.random.randn(dimension)
    bias = np.random.rand()
    noise_std = np.random.normal(0, noise_std, size=size)

    # create y (Add noise to multi-collinearity)
    y = X.dot(weights) + bias + (bias + noise_std)

    # Split dataset into train and test
    train_X, train_y, test_X, test_y = split_dataset(X, y, test_ratio)
    return X, y, train_X, train_y, test_X, test_y


def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float) -> tuple:
    """
    split_dataset()
    This function splits the dataset into training and test sets
    :param X: Dataset with features
    :param y: Dataset labels
    :param test_ratio: size of the test set
    :return:
        train_X: train dataset
        train_y: train dataset label
        test_X: test dataset
        test_y: test dataset label
    """
    # Split data into train and test
    test_size = int(test_ratio * X.shape[0])
    train_X, test_X = X[:test_size], X[test_size:]
    train_y, test_y = y[:test_size], y[test_size:]
    return train_X, train_y, test_X, test_y


def get_data(data_type: str, args: dict) -> tuple:
    """
    get_data()
    This function loads dataset chosen data type
    :param data_type: the type of data to use [data options] 'iris'(default), 'file' ,'generate'
    :param args: a parameter setting of the dataset
    :returns: tuples produced by chosen function. Specific details as below:
        X: Dataset with features
        y: Dataset with labels
        train_X: train dataset
        train_y: train dataset label
        test_X: test dataset
        test_y: test dataset label
    """
    if data_type == 'file':
        return read_csv(**args)
    else:  # (Default) Generate data
        return generate_data(**args)


"""
    Metric functions
"""


def MSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE()
    This function calculates a value based on mean squared error (MSE),
    one of the evaluation indicators for regression models.
    :param y: desired target values
    :param y_pred: predicted values
    :return: mean squared error of the given data
    """
    return float(np.mean((y - y_pred) ** 2))


def accuracy_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    accuracy_rate()
    This function calculates a value based on accuracy rate,
    which is one of the evaluation metrics for a classification model.
    :param y: desired target values
    :param y_pred: predicted values
    :return: accuracy rate of the model
    """
    return float(np.sum(y == y_pred) / len(y))


def AIC(y: np.ndarray, X: np.ndarray, y_pred: np.ndarray):
    """
    aic()
    This function computes the AIC evaluation for the given model
    :param y: Actual values
    :param X: Feature vectors
    :param y_pred: Predicted values
    :return: AIC value
    """
    # Number of samples
    n = X.shape[0]

    # Number of feature including a residual
    k = X.shape[1] + 1

    # Log likelihood formula: log_likelihood = - n/2*log(2*π) - n/2*log(MSE)  - n/2
    log_likelihood = - n / 2 * np.log(2 * np.pi) - n / 2 - n * np.log(MSE(y, y_pred)) / 2

    return 2 * k - 2 * log_likelihood  # AIC = 2*k - 2*log_likelihood


def R2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R2()
    This function calculates the coefficient of determination (R2) between given data (actual y and predicted y).
    The higher the value of R², the higher the explanatory power of the model.
    Formula used:
        R² = 1 - (∑((y_i - y_pred_i)²) / ∑((y_i - y_mean)²))
    where:
        y_i      : actual weight of ith feature
        y_pred_i : predicted weight of ith feature
        y_mean   : mean of the actual weight features

    :param y: Actual weights
    :param y_pred: Predicted weights
    :return: R2 score between two weights
    """
    # Average of actual weights
    y_mean = np.mean(y)
    r_2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)  # R2 = 1 - SSR/SST
    return float(r_2)
