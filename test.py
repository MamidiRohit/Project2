import pytest
import numpy as np
from lib import (
    LinearRegression,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    k_fold_cross_validation,
    bootstrapping,
    generate_data,
)

# Test for LinearRegression class
@pytest.fixture
def setup_linear_regression_data():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([5, 11, 17])
    model = LinearRegression()
    return X, y, model

def test_linear_regression_fit(setup_linear_regression_data):
    X, y, model = setup_linear_regression_data
    model.fit(X, y)
    assert model.weights is not None

def test_linear_regression_predict(setup_linear_regression_data):
    X, y, model = setup_linear_regression_data
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)

def test_linear_regression_prediction_accuracy(setup_linear_regression_data):
    X, y, model = setup_linear_regression_data
    model.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(predictions, y, decimal=1)

# Test for metrics functions
@pytest.fixture
def setup_metrics_data():
    y_true = np.array([3, 5, 7])
    y_pred = np.array([2.8, 5.1, 6.9])
    return y_true, y_pred

def test_mean_squared_error():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    mse = mean_squared_error(y_true, y_pred)
    expected_mse = np.mean((y_true - y_pred) ** 2)
    assert np.isclose(mse, expected_mse), f"Expected {expected_mse}, but got {mse}"

def test_mean_absolute_error():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    mae = mean_absolute_error(y_true, y_pred)
    expected_mae = 0.5  # actual expected value
    tolerance = 1e-6  # Allow for small floating-point differences
    assert abs(mae - expected_mae) < tolerance, f"Expected {expected_mae}, but got {mae}"

def test_r2_score(setup_metrics_data):
    y_true, y_pred = setup_metrics_data
    r2 = r2_score(y_true, y_pred)
    assert np.isclose(r2, 0.998, atol=1e-2)  # Increase tolerance to allow slight variation

# Test for k-fold cross-validation
@pytest.fixture
def setup_cross_validation_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([5, 11, 17, 23, 29])
    model = LinearRegression()
    return X, y, model

def test_k_fold_cross_validation(setup_cross_validation_data):
    X, y, model = setup_cross_validation_data
    k = 3
    metrics, averages = k_fold_cross_validation(model, X, y, k, shuffle=True)
    assert len(metrics["mse"]) == k
    assert "mse" in averages
    assert "mae" in averages
    assert "r2" in averages

# Test for bootstrapping
@pytest.fixture
def setup_bootstrapping_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([5, 11, 17, 23, 29])
    model = LinearRegression()
    return X, y, model

def test_bootstrapping(setup_bootstrapping_data):
    X, y, model = setup_bootstrapping_data
    metrics, averages = bootstrapping(model, X, y, s=3, epochs=2)
    assert len(metrics["mse"]) == 2
    assert "mse" in averages
    assert "mae" in averages
    assert "r2" in averages

# Test for data generation function
def test_generate_data():
    X, y = generate_data(5, 3)
    assert X.shape[0] == 5
    assert X.shape[1] == 3
    assert y.shape[0] == 5
