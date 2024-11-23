# run_model.py

from Models.LinearRegressionModel import LinearRegression, RegressionMetrics
from Models.kfold import KFoldCrossValidation
from Models.bootstrapping import BootstrapModelSelection
from Data.Data_generator import SyntheticDataGenerator
import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    Assumes the target column is 'y' and all other columns are features.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Debugging: Print column names to verify
    print(f"Columns in {file_path}: {df.columns.tolist()}")  # Print column names for verification
    
    # Define the target column
    target_column = 'y'  # Update target column name based on your CSV
    
    # Validate if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the CSV file.")
    
    # Extract features (X) and target (y)
    X = df.drop(columns=[target_column]).to_numpy()  # Drop 'y' column to get features
    y = df[target_column].to_numpy()  # Extract 'y' as target variable
    
    return X, y


def run_linear_regression(X, y):
    """ Run linear regression model """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = RegressionMetrics.mean_squared_error(y, y_pred)
    r2 = RegressionMetrics.r_squared(y, y_pred)
    print(f"Linear Regression - MSE: {mse:.4f}, R-Squared: {r2:.4f}")
    return mse, r2

def run_bootstrapping(X, y):
    bootstrap = BootstrapModelSelection(n_iterations=5)
    bootstrap_results = bootstrap.evaluate(X, y)
    print(f"Bootstrapping - Mean MSE: {bootstrap_results['mean_mse']:.4f}, "
          f"Mean R-Squared: {bootstrap_results['mean_r2']:.4f}, "
          f"Mean AIC: {bootstrap_results['mean_aic']:.4f}, "
          f"Mean MAE: {bootstrap_results['mean_mae']:.4f}, "
          f"Mean RMSE: {bootstrap_results['mean_rmse']:.4f}")
    return bootstrap_results


def run_k_fold(X, y):
    kfold = KFoldCrossValidation(n_splits=5)
    kfold_results = kfold.evaluate(X, y)
    print(f"K-Fold - Mean MSE: {kfold_results['mean_mse']:.4f}, "
          f"Mean R-Squared: {kfold_results['mean_r2']:.4f}, "
          f"Mean AIC: {kfold_results['mean_aic']:.4f}, "
          f"Mean MAE: {kfold_results['mean_mae']:.4f}, "
          f"Mean RMSE: {kfold_results['mean_rmse']:.4f}")
    return kfold_results


if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = 'Data/test_data.csv' 

    # Load the data
    X, y = load_data(csv_file_path)

    # Run and evaluate models
    print("Running Linear Regression...")
    run_linear_regression(X, y)

    print("\nRunning K-Fold Cross Validation...")
    run_k_fold(X, y)

    print("\nRunning Bootstrapping...")
    run_bootstrapping(X, y)
