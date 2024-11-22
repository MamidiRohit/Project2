import numpy as np
import pandas as pd
from boosting.gradient_boosting import GradientBoostingTree
from model_selection.cross_validation import k_fold_cv
from model_selection.bootstrapping import bootstrap

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

    return X_scaled, y_scaled

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

if __name__ == "__main__":
    # 1. Load and Preprocess Dataset
    dataset_path = "data/boston_housing.csv"
    X, y = load_dataset(dataset_path)
    X, y = preprocess_data(X, y)

    # 2. Perform K-Fold Cross-Validation
    print("\nPerforming K-Fold Cross-Validation...")
    model = GradientBoostingTree(n_estimators=100, learning_rate=0.1, max_depth=3)
    cv_score = k_fold_cv(model, X, y, k=5, metric="mse")
    print(f"K-Fold Cross-Validation MSE: {cv_score:.4f}")

    # 3. Perform Bootstrapping
    print("\nPerforming Bootstrapping...")
    bootstrap_scores, mean_bootstrap_score = bootstrap(model, X, y, B=10, metric="mse")
    print(f"Bootstrap Mean MSE: {mean_bootstrap_score:.4f}")
    print(f"Bootstrap Scores: {bootstrap_scores}")

    # 4. Perform Grid Search for max_depth
    print("\nPerforming Grid Search for max_depth...")
    max_depth_values = [2, 3, 5]
    best_max_depth, best_cv_score = grid_search_max_depth(X, y, max_depth_values)
    print(f"Best Max Depth: {best_max_depth}")
    print(f"Best CV MSE: {best_cv_score:.4f}")

    # 5. Train Final Model with Best Parameters
    print("\nTraining final model with best parameters...")
    final_model = GradientBoostingTree(
        n_estimators=100, learning_rate=0.1, max_depth=best_max_depth
    )
    final_model.fit(X, y)
    print("Final model training complete!")
