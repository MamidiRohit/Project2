"""
References:
    1. AIC definition: Wikipedia - Akaike information criterion
      [LINK] https://en.wikipedia.org/wiki/Akaike_information_criterion
    2. Log likelihood formula - StatLect "Log-likelihood"
      [LINK] https://www.statlect.com/glossary/log-likelihood
    3. R2 score formula: Wikipedia - Coefficient of determination
      [LINK] https://en.wikipedia.org/wiki/Coefficient_of_determination
"""
import json
import numpy as np
from sklearn.linear_model import *

# Collection of auxiliary functions

"""
    Parameter setting read
"""


def get_param(file_path: str) -> dict:
    """
    get_param()
    This function reads parameter settings from a JSON file.
    :param file_path: Path to the JSON file containing parameter settings.
    :return: A dictionary of parameters.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


"""
    Model selection functions
"""


def get_model(model_type: str):
    """
    get_model()
    This function returns a model object based on the given model_type.
    If an invalid model_type is provided, it defaults to LinearRegression.
    :param model_type: The type of model to use. Options: "LinearRegression" (default), "LogisticRegression".
    :return: A model object.
    """
    print(f"Model Type: {model_type}")

    if model_type == "LinearRegression":
        return LinearRegression()
    elif model_type == "LogisticRegression":
        return LogisticRegression(max_iter=1000)
    else:  # Invalid type described
        print(f"[Warning] Invalid model type given. Defaulting to 'LinearRegression'.")
        return LinearRegression()


def get_metric(metric_type: str):
    """
    get_metric()
    This function returns the metric function corresponding to the given metric_type.
    Defaults to MSE if an invalid type is provided.
    :param metric_type: The type of metric to use. Options: "MSE" (default), "Accuracy score", "R2".
    :return: A metric function.
    """
    print(f"Metric Type: {metric_type}")

    if metric_type == "MSE":
        return MSE
    elif metric_type == "Accuracy score":
        return accuracy_score
    elif metric_type == "R2":
        return R2
    else:  # Invalid type described
        print(f"[Warning] Invalid metric type given. Defaulting to 'MSE'.")
        return MSE


"""
    Data import and generation functions
"""


def read_csv(file_path: str, test_ratio: float) -> tuple:
    """
    read_csv()
    This function reads data from a CSV file and splits it into training and test sets.
    :param file_path: The path of the CSV file to read.
    :param test_ratio: The proportion of data to use for the test set.
    :return: A tuple containing the full dataset (X, y) and the split datasets (train_X, train_y, test_X, test_y).
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Skip the header row

    X = data[:, :-1]
    y = data[:, -1]

    # Split dataset into train and test
    train_X, train_y, test_X, test_y = split_dataset(X, y, test_ratio)
    return X, y, train_X, train_y, test_X, test_y


def generate_data(size: int, dimension: int, correlation: float, noise_std: float, random_state: int,
                  test_ratio: float) -> tuple:
    """
    generate_data()
    This function generates synthetic data with multi-collinearity, designed for testing models like ElasticNet.
    :param size: Number of samples to generate.
    :param dimension: Number of features to generate.
    :param correlation: Correlation coefficient between features (1 indicates perfect correlation).
    :param noise_std: Standard deviation of noise added to the data.
    :param random_state: Random seed for reproducibility.
    :param test_ratio: The proportion of data to use for the test set.
    :return: A tuple containing the full dataset (X, y) and the split datasets (train_X, train_y, test_X, test_y).
    """
    if random_state:
        np.random.seed(random_state)

    # TODO: Update Multi-collinearity function part
    # Generate the base feature
    X_base = np.random.rand(size, 1)

    # Create correlated features
    X = X_base + correlation * np.random.randn(size, dimension) * noise_std

    # Increase the correlation between each feature (e.g. through linear combination)
    for i in range(1, dimension):
        X[:, i] = correlation * X_base[:, 0] + (1 - correlation) * np.random.randn(size)

    # Generate weights, bias, and noise
    weights = np.random.randn(dimension)
    bias = np.random.rand()
    noise_std = np.random.normal(0, noise_std, size=size)

    # Create the target variable y
    y = X.dot(weights) + bias + (bias + noise_std)

    # Split dataset into train and test
    train_X, train_y, test_X, test_y = split_dataset(X, y, test_ratio)
    return X, y, train_X, train_y, test_X, test_y


def split_dataset(X: np.ndarray, y: np.ndarray, test_ratio: float) -> tuple:
    """
    split_dataset()
    This function splits the dataset into training and test sets based on the given test_ratio.
    :param X: Features.
    :param y: Labels.
    :param test_ratio: Proportion of the dataset to use as the test set.
    :return: Training and test sets (train_X, train_y, test_X, test_y).
    """
    # Split data into train and test
    test_size = int(test_ratio * X.shape[0])
    train_X, test_X = X[:test_size], X[test_size:]
    train_y, test_y = y[:test_size], y[test_size:]
    return train_X, train_y, test_X, test_y


def get_data(data_type: str, args: dict) -> tuple:
    """
    get_data()
    This function loads or generates a dataset based on the data_type.
    :param data_type: The type of dataset to use. Options: "file" (from CSV) or "generate" (synthetic data).
    :param args: Arguments specific to the chosen data type.
    :return: Dataset tuples (X, y, train_X, train_y, test_X, test_y).
    """
    if data_type == 'file':
        return read_csv(**args)
    else:  # Default to generated data
        return generate_data(**args)


"""
    Metric functions
"""


def MSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE()
    This function calculates the Mean Squared Error (MSE) between actual and predicted values.
    :param y: Actual values.
    :param y_pred: Predicted values.
    :return: Mean Squared Error.
    """
    return float(np.mean((y - y_pred) ** 2))


def accuracy_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    accuracy_score()
    This function calculates the accuracy score for classification tasks.
    :param y: Actual labels.
    :param y_pred: Predicted labels.
    :return: Accuracy score.
    """
    return float(np.sum(y == y_pred) / len(y))


def AIC(y: np.ndarray, X: np.ndarray, y_pred: np.ndarray):
    """
    AIC()
    This function computes the Akaike Information Criterion (AIC) for the given model.
    Formula used:
        * Refer to above references
        log-likelihood = - n/2*log(2*π) - n/2*log(MSE)  - n/2
        AIC = 2*k - 2*(log-likelihood)
            = - (∑((y_i - y_pred_i)²) / ∑((y_i - y_mean)²))
    :param y: Actual values.
    :param X: Feature matrix.
    :param y_pred: Predicted values.
    :return: AIC value.
    """
    # Number of samples and features
    n = X.shape[0]
    k = X.shape[1] + 1  # Include residual as a parameter

    # compute Log-likelihood
    log_likelihood = - n / 2 * np.log(2 * np.pi) - n / 2 - n * np.log(MSE(y, y_pred)) / 2

    # Return AIC value
    return 2 * k - 2 * log_likelihood


def R2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R2()
    This function calculates the R² score (coefficient of determination) for regression models.
    Formula used:
        R² = 1 - (∑((y_i - y_pred_i)²) / ∑((y_i - y_mean)²))
    where:
        y_i      : actual weight of ith feature
        y_pred_i : predicted weight of ith feature
        y_mean   : mean of the actual weight features
    :param y: Actual values.
    :param y_pred: Predicted values.
    :return: R² score.
    """
    # Average of actual weights
    y_mean = np.mean(y)
    r_2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)  # R2 = 1 - SSR/SST
    return float(r_2)


"""
    Data write & Visualization
"""


def visualize(data: np.ndarray):
    """TODO Create this function"""
    pass


def write_result(file_path: str, data: np.ndarray):
    """TODO Create this function"""
    pass
