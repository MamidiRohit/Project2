from models.gradient_boosting_tree import GradientBoostingTree
from utils.data_loader import load_data
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return mse, r2, predictions

if __name__ == "__main__":
    # Load data
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    X_train, X_test, y_train, y_test = load_data(url, target_column="medv")

    # Hyperparameter options
    learning_rates = [0.05, 0.01]
    n_estimators_options = [100, 200]
    max_depths = [3, 4, 5]
    min_samples_split = 10

    best_mse = float("inf")
    best_params = {}
    all_results = []

    # Train models with different hyperparameters
    for learning_rate in learning_rates:
        for n_estimators in n_estimators_options:
            for max_depth in max_depths:
                # Initialize the model
                model = GradientBoostingTree(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                
                # Fit the model
                model.fit(X_train, y_train)

                # Evaluate on training data
                train_mse, train_r2, train_predictions = evaluate_model(model, X_train, y_train)

                # Evaluate on test data
                test_mse, test_r2, test_predictions = evaluate_model(model, X_test, y_test)

                # Store results
                all_results.append({
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "train_mse": train_mse,
                    "train_r2": train_r2,
                    "test_mse": test_mse,
                    "test_r2": test_r2,
                    "train_predictions": train_predictions,
                    "test_predictions": test_predictions
                })

                # Track the best model based on test MSE
                if test_mse < best_mse:
                    best_mse = test_mse
                    best_params = {
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth
                    }

    # Print results for each parameter combination
    for result in all_results:
        print(f"Learning Rate: {result['learning_rate']}, N Estimators: {result['n_estimators']}, Max Depth: {result['max_depth']}")
        print(f"Train MSE: {result['train_mse']}, Train R²: {result['train_r2']}")
        print(f"Test MSE: {result['test_mse']}, Test R²: {result['test_r2']}\n")

    # Output best hyperparameters and performance
    print("\nBest Parameters:", best_params)
    print("Best Test Mean Squared Error:", best_mse)

    # Plot results for MSE
    mse_results_df = pd.DataFrame(all_results)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group_df in mse_results_df.groupby("learning_rate"):
        group_df.plot(x="max_depth", y="test_mse", ax=ax, label=f"Learning Rate = {label}")
    ax.set_title("Test MSE vs Max Depth for Different Learning Rates")
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.show()

    # Final evaluation on the best model
    best_model = GradientBoostingTree(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        min_samples_split=min_samples_split
    )
    best_model.fit(X_train, y_train)

    # Evaluate final model on training and test data
    final_train_mse, final_train_r2, final_train_predictions = evaluate_model(best_model, X_train, y_train)
    final_test_mse, final_test_r2, final_test_predictions = evaluate_model(best_model, X_test, y_test)

    print("\nFinal Model Evaluation on Training Data:")
    print(f"Train MSE: {final_train_mse}")
    print(f"Train R²: {final_train_r2}")
    print(f"Train Predictions: {final_train_predictions[:10]}")

    print("\nFinal Model Evaluation on Test Data:")
    print(f"Test MSE: {final_test_mse}")
    print(f"Test R²: {final_test_r2}")
    print(f"Test Predictions: {final_test_predictions[:10]}")
