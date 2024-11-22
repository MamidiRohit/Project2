import numpy as np
import pandas as pd
from boosting.gradient_boosting import GradientBoostingTree
from model_selection.cross_validation import k_fold_cv
from model_selection.bootstrapping import bootstrap
from sklearn.ensemble import GradientBoostingRegressor

def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Dataset is empty. Please provide a valid dataset.")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

def preprocess_data(X, y):
    """
    Preprocess the dataset: scale features and return scaled X and y.
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std

    return X_scaled, y_scaled, y_mean, y_std

def grid_search_max_depth(X, y, max_depth_values, n_estimators=100, learning_rate=0.1, k=5):
    """
    Perform grid search to find the best max_depth using k-fold cross-validation.
    """
    best_score = float("inf")
    best_max_depth = None

    for max_depth in max_depth_values:
        model = GradientBoostingTree(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        score = k_fold_cv(model, X, y, k=k, metric="mse")
        print(f"Max Depth: {max_depth}, CV MSE: {score:.4f}")
        if score < best_score:
            best_score = score
            best_max_depth = max_depth

    return best_max_depth, best_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using MSE, MAE, and R² metrics.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, mae, r2

def plot_comparison(y_true, custom_pred, sklearn_pred):
    """
    Plot comparison of predictions from both models against true values.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, custom_pred, alpha=0.7, label="Custom Model Predictions", color="blue")
    plt.scatter(y_true, sklearn_pred, alpha=0.7, label="Sklearn Model Predictions", color="green")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Comparison of Custom vs Sklearn Model Predictions")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Load and Preprocess Dataset
    dataset_path = "data/highly_correlated_dataset.csv"
    X, y = load_dataset(dataset_path)
    X, y, y_mean, y_std = preprocess_data(X, y)

    # 2. Perform K-Fold Cross-Validation for Custom Model
    print("\nPerforming K-Fold Cross-Validation for Custom Model...")
    custom_model = GradientBoostingTree(n_estimators=100, learning_rate=0.1, max_depth=3)
    custom_cv_score = k_fold_cv(custom_model, X, y, k=5, metric="mse")
    print(f"Custom Model K-Fold Cross-Validation MSE: {custom_cv_score:.4f}")

    # 3. Perform K-Fold Cross-Validation for Sklearn Model
    print("\nPerforming K-Fold Cross-Validation for Sklearn Model...")
    sklearn_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    sklearn_cv_score = k_fold_cv(sklearn_model, X, y, k=5, metric="mse")
    print(f"Sklearn Model K-Fold Cross-Validation MSE: {sklearn_cv_score:.4f}")

    # 4. Perform Bootstrapping for Custom Model
    print("\nPerforming Bootstrapping for Custom Model...")
    custom_bootstrap_scores, custom_mean_bootstrap_score = bootstrap(custom_model, X, y, B=10, metric="mse")
    print(f"Custom Model Bootstrap Mean MSE: {custom_mean_bootstrap_score:.4f}")
    print(f"Custom Model Bootstrap Scores: {custom_bootstrap_scores}")

    # 5. Perform Bootstrapping for Sklearn Model
    print("\nPerforming Bootstrapping for Sklearn Model...")
    sklearn_bootstrap_scores, sklearn_mean_bootstrap_score = bootstrap(sklearn_model, X, y, B=10, metric="mse")
    print(f"Sklearn Model Bootstrap Mean MSE: {sklearn_mean_bootstrap_score:.4f}")
    print(f"Sklearn Model Bootstrap Scores: {sklearn_bootstrap_scores}")

    # 6. Perform Grid Search for Best Max Depth for Custom Model
    print("\nPerforming Grid Search for Best Max Depth for Custom Model...")
    max_depth_values = [2, 3, 5]
    best_max_depth, best_cv_score = grid_search_max_depth(X, y, max_depth_values)
    print(f"Best Max Depth: {best_max_depth}")
    print(f"Best Custom Model CV MSE: {best_cv_score:.4f}")

    # 7. Train Final Custom Model with Best Parameters
    print("\nTraining Final Custom Model with Best Parameters...")
    final_custom_model = GradientBoostingTree(
        n_estimators=100, learning_rate=0.1, max_depth=best_max_depth
    )
    final_custom_model.fit(X, y)
    print("Final Custom Model training complete!")

    # 8. Train Final Sklearn Model
    print("\nTraining Final Sklearn Model...")
    final_sklearn_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=best_max_depth, random_state=42
    )
    final_sklearn_model.fit(X, y)
    print("Final Sklearn Model training complete!")

    # 9. Make Predictions for Final Models
    custom_pred = final_custom_model.predict(X) * y_std + y_mean
    sklearn_pred = final_sklearn_model.predict(X) * y_std + y_mean
    y_original = y * y_std + y_mean

    # 10. Evaluate Both Models
    print("\nEvaluating Both Models...")
    custom_mse, custom_mae, custom_r2 = evaluate_model(y_original, custom_pred)
    print(f"Custom Model - MSE: {custom_mse:.4f}, MAE: {custom_mae:.4f}, R²: {custom_r2:.4f}")

    sklearn_mse, sklearn_mae, sklearn_r2 = evaluate_model(y_original, sklearn_pred)
    print(f"Sklearn Model - MSE: {sklearn_mse:.4f}, MAE: {sklearn_mae:.4f}, R²: {sklearn_r2:.4f}")

    # 11. Plot Comparison of Both Models
    print("\nPlotting Comparison of Both Models...")
    plot_comparison(y_original, custom_pred, sklearn_pred)