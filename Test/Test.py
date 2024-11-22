import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.LinearRegression import LinearRegressionModel, Metrics
from Models.Kfold import CrossVal
from Models.Bootstrapping import Bootstrapping
from Data.DataGen import DataGenerator, ProfessorData


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an Elastic Net Model on Generated Data")
    
    # Data generation arguments
    parser.add_argument('--rows', type=int, help="Number of Rows/Samples in Generated Data", default=100)
    parser.add_argument('--cols', type=int, help="Number of Columns/Features in Generated Data", default=10)
    parser.add_argument('--noise', type=float, help="Noise Scale in Generated Data", default=0.5)
    parser.add_argument('--seed', type=int, help="Random Seed for Data Generation", default=42)
    
    # Professor data generation arguments
    parser.add_argument('-N', type=int, help="Number of samples.", default=100)
    parser.add_argument('-m', nargs='+', type=float, help="Expected regression coefficients", default=[1, -2, 3])
    parser.add_argument('-b', type=float, help="Offset", default=5.0)
    parser.add_argument('-scale', type=float, help="Scale of noise", default=0.5)
    parser.add_argument('-rnge', nargs=2, type=float, help="Range of Xs", default=[-10, 10])
    parser.add_argument('-random_seed', type=int, help="A seed to control randomness", default=42)
    
    # Model evaluation arguments
    parser.add_argument('-k', type=int, help="Number of k-folds.", default=5)
    parser.add_argument('-n_iter', type=int, help="N_Iterations for Bootstrapping.", default=10)
    
    # CSV input arguments
    parser.add_argument('--csv_file_path', type=str, help="CSV file path")
    parser.add_argument('--target_column', type=str, help="Name of the target column in the CSV file")
    
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
    
    elif args.rows and args.cols and args.noise:

        data_gen = DataGenerator(rows=args.rows, cols=args.cols, noise=args.noise, seed=args.seed)
        X, y = data_gen.gen_data()
        X, y = X.values, y.values  
    
    else:

        prof_data_gen = ProfessorData(
            rnge=args.rnge,
            scale=args.scale, 
            m=args.m,
            b=args.b,
            N=args.N,
            random_seed=args.random_seed
        )
        X, y = prof_data_gen.linear_data_generator()

  
    cv = CrossVal(k=args.k)
    mse_cv, r2_cv, aic_cv = cv.kFold(X, y)
    print(f"\nAverage k-Fold CV MSE: {mse_cv:.4f}")
    print(f"Average k-Fold CV R-Squared: {r2_cv:.4f}")
    print(f"Average k-Fold CV Aic Score: {aic_cv:.4f}")

  
    print("\nBootstrapping Results:")
    bs_model = Bootstrapping(n_iter=args.n_iter)
    mse_bs, r2_bs, aic_bs = bs_model.bootstrap(X, y)
    print(f"\nAverage Bootstrap MSE: {mse_bs:.4f}")
    print(f"Average Bootstrap R-Squared: {r2_bs:.4f}")
    print(f"Average Bootstrap Aic Score: {aic_bs:.4f}")
