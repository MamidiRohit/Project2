
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.LinearRegression import LinearRegressionModel, Metrics
from Models.Kfold import CrossVal
from Models.Bootstrapping import Bootstrapping
from Data.DataGen import DataGenerator, ProfessorData


def test_linear_regression():
    generator = DataGenerator(rows=100, cols=5, noise=0.5)
    X , y = generator.gen_data()
    model = LinearRegressionModel()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = Metrics.mean_squared_error(y, y_pred)
    r2 = Metrics.r_squared(y, y_pred)
    assert mse < 1.0, "MSE is too high"
    assert 0.9 <= r2 <= 1.0, "R-Squared is out of expected range"

def test_k_fold():
    generator = ProfessorData(m=[2, -1, 3, 0, 0], N=100, b=0, scale=0.5)
    X, y = generator.linear_data_generator()
    cv = CrossVal(k=5)
    mse, r2, aic = cv.kFold(X, y)
    assert mse < 1.0, "MSE from k-fold cross-validation is too high"
    assert 0.8 <= r2 <= 1.0, "R-Squared from k-fold cross-validation is out of expected range"

def test_bootstrapping():
    generator = ProfessorData(m=[2, -1, 3, 0, 0], N=100, b=0, scale=0.5)
    X, y = generator.linear_data_generator()
    bootstrap = Bootstrapping(n_iter=10)
    mse, r2, aic= bootstrap.bootstrap(X, y)
    assert mse < 1.0, "MSE from bootstrapping is too high"
    assert 0.8 <= r2 <= 1.0, "R-Squared from bootstrapping is out of expected range"

def test_data_generation():
    generator = DataGenerator(rows=100, cols=5, noise=0.5)
    X, y = generator.gen_data()
    assert X.shape == (100, 5), "Data generation did not produce expected feature dimensions"
    assert len(y) == 100, "Data generation did not produce expected target length"

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    return X, y

def test_csv_input(tmp_path):
    file_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "feature1": np.random.rand(2000),
        "feature2": np.random.rand(2000),
        "target": np.random.rand(2000)
    })
    df.to_csv(file_path, index=False)
    loaded_df = pd.read_csv(file_path)
    assert "feature1" in loaded_df.columns, "CSV should contain 'feature1' column."
    assert "target" in loaded_df.columns, "CSV should contain 'target' column."
    assert len(loaded_df) == 2000, "CSV should contain 2000 rows."
