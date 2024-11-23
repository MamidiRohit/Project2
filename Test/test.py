import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.LinearRegressionModel import LinearRegression, RegressionMetrics
from Models.kfold import KFoldCrossValidation
from Models.bootstrapping import BootstrapModelSelection
from Data.Data_generator import SyntheticDataGenerator, CustomDataGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an Elastic Net Model on Generated Data")
    
    # CSV input arguments (group them together)
    parser.add_argument('--csv_file_path', type=str, help="CSV file path")
    parser.add_argument('--target_column', type=str, help="Name of the target column in the CSV file")
    
    # Model evaluation arguments (group together)
    parser.add_argument('-k', type=int, help="Number of k-folds.", default=5)
    parser.add_argument('-n_iter', type=int, help="N_Iterations for Bootstrapping.", default=10)
    
    # Custom data generation arguments (group together)
    parser.add_argument('-N', type=int, help="Number of samples.", default=100)
    parser.add_argument('-m', nargs='+', type=float, help="Expected regression coefficients", default=[1, -2, 3])
    parser.add_argument('-b', type=float, help="Offset", default=5.0)
    parser.add_argument('-scale', type=float, help="Scale of noise", default=0.5)
    parser.add_argument('-rnge', nargs=2, type=float, help="Range of Xs", default=[-10, 10])
    parser.add_argument('-random_seed', type=int, help="A seed to control randomness", default=42)
    
    # Data generation arguments (group together)
    parser.add_argument('--rows', type=int, help="Number of Rows/Samples in Generated Data", default=100)
    parser.add_argument('--cols', type=int, help="Number of Columns/Features in Generated Data", default=10)
    parser.add_argument('--noise_level', type=float, help="Noise Scale in Generated Data", default=0.5)
    
    # Optional Arguments (group together)
    parser.add_argument('--random_seed', type=int, help="Random Seed for Data Generation", default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Load data based on input arguments
    if args.csv_file_path and args.target_column:
        df = pd.read_csv(args.csv_file_path)
        print("Columns in DataFrame:", df.columns.tolist())

        if args.target_column in df.columns:
            X = df.drop(args.target_column, axis=1).values
            y = df[args.target_column].values
        else:
            raise KeyError(f"Column '{args.target_column}' not found in DataFrame.") 
    
    elif args.rows and args.cols and args.noise_level:
        data_gen = SyntheticDataGenerator(rows=args.rows, cols=args.cols, noise_level=args.noise_level, random_seed=args.random_seed)
        X, y = data_gen.generate_data()
    
    else:
        cust_data_gen = CustomDataGenerator(
            rnge=args.rnge,
            scale=args.scale, 
            m=args.m,
            b=args.b,
            N=args.N,
            random_seed=args.random_seed
        )
        X, y = cust_data_gen.linear_data_generator()

    # KFold Cross Validation
    cv = KFoldCrossValidation(n_splits=args.k)
    results_cv = cv.evaluate(X, y)

    mse_cv = float(results_cv["mean_mse"])
    r2_cv = float(results_cv["mean_r2"])
    aic_cv = float(results_cv["mean_aic"])
    mae_cv = float(results_cv["mean_mae"])
    rmse_cv = float(results_cv["mean_rmse"])

    print(f"\nAverage k-Fold CV MSE: {mse_cv:.4f}")
    print(f"Average k-Fold CV R-Squared: {r2_cv:.4f}")
    print(f"Average k-Fold CV AIC Score: {aic_cv:.4f}")
    print(f"Average k-Fold CV MAE: {mae_cv:.4f}")
    print(f"Average k-Fold CV RMSE: {rmse_cv:.4f}")

    # Bootstrapping Evaluation
    print("\nBootstrapping Results:")
    bs_model = BootstrapModelSelection(n_iterations=args.n_iter)
    results_bs = bs_model.evaluate(X, y)

    mse_bs = float(results_bs["mean_mse"])
    r2_bs = float(results_bs["mean_r2"])
    aic_bs = float(results_bs["mean_aic"])
    mae_bs = float(results_bs["mean_mae"])
    rmse_bs = float(results_bs["mean_rmse"])

    print(f"\nAverage Bootstrap MSE: {mse_bs:.4f}")
    print(f"Average Bootstrap R-Squared: {r2_bs:.4f}")
    print(f"Average Bootstrap AIC Score: {aic_bs:.4f}")
    print(f"Average Bootstrap MAE: {mae_bs:.4f}")
    print(f"Average Bootstrap RMSE: {rmse_bs:.4f}")
