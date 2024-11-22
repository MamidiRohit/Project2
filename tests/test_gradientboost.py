import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from models.checker import *
from models.GradientBoost import GradientBoostingTree
from models.gridsearch import *


def test_predict():   
    """
    Function to load a dataset, preprocess data, perform hyperparameter tuning,
    train a Gradient Boosting model, evaluate its performance, and visualize results.

    Steps:
    1. Prompts the user for a dataset file path and loads it (supports .csv, .xlsx, .json, .parquet).
    2. Displays a preview of the dataset and checks for null values.
    3. Splits the dataset into training and testing sets (80%-20% split).
    4. Normalizes the feature columns to the range [0, 1].
    5. Performs grid search to tune hyperparameters (n_estimators, learning_rate, max_depth).
    6. Fits a Gradient Boosting model using the best parameters.
    7. Evaluates the model using R² Score, MAE, and RMSE.
    8. Visualizes the results with density and scatter plots for predicted vs actual values.

    Returns:
    - None. Prints evaluation metrics and displays plots.
    
    """
    file_path = input("Please enter the path to your dataset file: ")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print(
                "Unsupported file format. Please provide a CSV, Excel, JSON, or Parquet file.")
            return
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return

    print("\n" + "=" * 40)
    print("Dataset Preview:")
    print("=" * 40)
    print(df.head())

    target = input("Enter the target column name: ")

    check_null(df)

    X, Y = XandY(df, target)

    np.random.seed(42)

    shuffled_indices = np.random.permutation(X.shape[0])

    train_size = int(0.8 * len(shuffled_indices))
    train_indices, test_indices = shuffled_indices[:
                                                   train_size], shuffled_indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = Y[train_indices], Y[test_indices]
    
    #! Normalization
    print("\nApplying the Normalization for the featured columns")
    X_train_normalized, train_min, train_max = normalize_data(X_train)
    X_test_normalized, _, _ = normalize_data(X_test)
    print("\nPre-processing is Done")
    
    #! Hyper-parameter Tunning
     
    n_estimators_values = [10, 50, 100]
    learning_rate_values = [0.1, 0.01, 0.001]
    max_depth_values = [2, 3, 5]

    print("\nUsing Grid Search to find the best parametrs")
    print("\nDepends on the CPU/GPU power it will take time  ")
    print("\nplease wait................................")
    best_params, best_score = grid_search_gradient_boosting(
        X_train_normalized, y_train, X_test_normalized, y_test,
        n_estimators_values, learning_rate_values, max_depth_values
    )

    print("\n" + "=" * 40)
    print("Best Parameters from Grid Search")
    print("=" * 40)
    print(f"n_estimators: {best_params['n_estimators']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Max Depth: {best_params['max_depth']}")
    print("\n" + "=" * 40)
    print(f"Best R² score: {best_score:.4f}")

    final_model = GradientBoostingTree(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"]
    )
    
    #! Fitting with Best parameters
    final_model.fit(X_train_normalized, y_train)
    y_pred_final = final_model.predict(X_test_normalized)

    r2_final = final_model.r2_score_manual(y_test, y_pred_final)
    mae_final = final_model.mae_manual(y_test, y_pred_final)
    rmse_final = final_model.rmse_manual(y_test, y_pred_final)
    
    
    #! Accuracy and model Evaluations 

    print("\nFinal Model Evaluation")
    print("=" * 40)
    print(f"R² Score: {r2_final:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_final:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_final:.4f}")

    y_test = np.array(y_test).ravel()
    y_pred_final = np.array(y_pred_final).ravel()
    
    
    #! PLOTS 

    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test, color='blue', fill=True, label='Actual Values')
    sns.kdeplot(y_pred_final, color='green',
                fill=True, label='Predicted Values')
    plt.title('Density Plot of Actual vs Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_final, color='blue',
                label='Predicted Values', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

test_predict()
