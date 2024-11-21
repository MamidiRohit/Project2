import numpy as np
import pandas as pd
from Project2.model_selection.generic_crossvalidation import ModelSelector

def test_wine_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    dataset_name = "Test Wine Quality"

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    selector = ModelSelector()

    models = ["linear", "ridge", "lasso"]
    alpha = 1.0
    results = {}
    print("\nWine Quality Dataset Test")
    for model in models:
        k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)
        bootstrap_mse = selector.bootstrapping(X, y, num_samples=100, model_type=model, alpha=alpha)

        # Fit the model to calculate AIC
        if model == "linear":
            beta = selector.fit_linear_regression(X, y)
            y_pred_function = selector.predict_linear_regression
        elif model == "ridge":
            beta = selector.fit_ridge_regression(X, y, alpha)
            y_pred_function = selector.predict_ridge_regression
        elif model == "lasso":
            beta = selector.fit_lasso_regression(X, y, alpha)
            y_pred_function = selector.predict_lasso_regression

        aic_score = selector.aic(X, y, beta, model)

        # Print results
        print(f"\nModel: {model.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {k_fold_mse:.4f}")
        print(f"Bootstrapping MSE: {bootstrap_mse:.4f}")
        print(f"AIC Score: {aic_score:.4f}")

        results[model] = {
            "k_fold_mse": k_fold_mse,
            "bootstrap_mse": bootstrap_mse,
            "aic_score": aic_score
        }

    # Identify and print the best model
    selector.best_model(results)
    selector.visualize_model_comparison(results, dataset_name)

# ================================================================================================================================

def test_boston_housing():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    data = pd.read_csv(url)
    dataset_name = "Test Boston Housing"

    X = data.drop('medv', axis=1).values
    y = data['medv'].values

    selector = ModelSelector()

    models = ["linear", "ridge", "lasso"]
    alpha = 0.5

    results = {}
    print("\nBoston Housing Dataset Test")

    for model in models:
        k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)

        bootstrap_mse = selector.bootstrapping(X, y, num_samples=100, model_type=model, alpha=alpha)

        if model == "linear":
            beta = selector.fit_linear_regression(X, y)
        elif model == "ridge":
            beta = selector.fit_ridge_regression(X, y, alpha)
        elif model == "lasso":
            beta = selector.fit_lasso_regression(X, y, alpha)

        aic_score = selector.aic(X, y, beta, model)

        # Print results
        print(f"\nModel: {model.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {k_fold_mse:.4f}")
        print(f"Bootstrapping MSE: {bootstrap_mse:.4f}")
        print(f"AIC Score: {aic_score:.4f}")

        # Store results
        results[model] = {
            "k_fold_mse": k_fold_mse,
            "bootstrap_mse": bootstrap_mse,
            "aic_score": aic_score
        }

    # Identify and print the best model
    selector.best_model(results)
    selector.visualize_model_comparison(results,dataset_name)

# ========================================================================================================================

def test_diabetes():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    data = pd.read_csv(url, header=None)
    dataset_name = "Test Diabetes"

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    selector = ModelSelector()

    models = ["linear", "ridge", "lasso"]
    alpha = 0.1

    results = {}
    print("\nDiabetes Dataset Test")

    for model in models:
        k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)

        bootstrap_mse = selector.bootstrapping(X, y, num_samples=150, model_type=model, alpha=alpha)

        if model == "linear":
            beta = selector.fit_linear_regression(X, y)
        elif model == "ridge":
            beta = selector.fit_ridge_regression(X, y, alpha)
        elif model == "lasso":
            beta = selector.fit_lasso_regression(X, y, alpha)

        aic_score = selector.aic(X, y, beta, model)

        # Print results
        print(f"\nModel: {model.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {k_fold_mse:.4f}")
        print(f"Bootstrapping MSE: {bootstrap_mse:.4f}")
        print(f"AIC Score: {aic_score:.4f}")

        # Store results
        results[model] = {
            "k_fold_mse": k_fold_mse,
            "bootstrap_mse": bootstrap_mse,
            "aic_score": aic_score
        }

    # Identify and print the best model
    selector.best_model(results)
    selector.visualize_model_comparison(results, dataset_name)

# ========================================================================================================================

def test_concrete_strength():
    url = "Concrete_Data.csv"
    data = pd.read_csv(url)
    dataset_name = "Test Concrete Strength"

    X = data.drop('Concrete compressive strength(MPa, megapascals) ', axis=1).values
    y = data['Concrete compressive strength(MPa, megapascals) '].values

    selector = ModelSelector()

    models = ["linear", "ridge", "lasso"]
    alpha = 0.1

    results = {}
    print("\nConcrete Compressive Strength Dataset Test")

    for model in models:
        k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)

        bootstrap_mse = selector.bootstrapping(X, y, num_samples=150, model_type=model, alpha=alpha)

        if model == "linear":
            beta = selector.fit_linear_regression(X, y)
        elif model == "ridge":
            beta = selector.fit_ridge_regression(X, y, alpha)
        elif model == "lasso":
            beta = selector.fit_lasso_regression(X, y, alpha)

        aic_score = selector.aic(X, y, beta, model)

        # Print results
        print(f"\nModel: {model.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {k_fold_mse:.4f}")
        print(f"Bootstrapping MSE: {bootstrap_mse:.4f}")
        print(f"AIC Score: {aic_score:.4f}")

        # Store results
        results[model] = {
            "k_fold_mse": k_fold_mse,
            "bootstrap_mse": bootstrap_mse,
            "aic_score": aic_score
        }

    # Identify and print the best model
    selector.best_model(results)
    selector.visualize_model_comparison(results, dataset_name)

# ========================================================================================================================

def test_polynomial_regression():

    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 3 + 1.5 * X.squeeze() - 0.5 * (X.squeeze() ** 2) + np.random.randn(100) * 5
    dataset_name = "Test Polynomial Regression"

    degree = 2
    X_poly = np.hstack([X ** i for i in range(1, degree + 1)])

    selector = ModelSelector()

    models = ["linear", "ridge", "lasso"]
    alpha = 1.0

    results = {}
    print("\nPolynomial Regression Test")

    for model in models:
        k_fold_mse = selector.k_fold_cross_validation(X_poly, y, k=5, model_type=model, alpha=alpha)

        bootstrap_mse = selector.bootstrapping(X_poly, y, num_samples=100, model_type=model, alpha=alpha)

        if model == "linear":
            beta = selector.fit_linear_regression(X_poly, y)
            y_pred_function = selector.predict_linear_regression
        elif model == "ridge":
            beta = selector.fit_ridge_regression(X_poly, y, alpha)
            y_pred_function = selector.predict_ridge_regression
        elif model == "lasso":
            beta = selector.fit_lasso_regression(X_poly, y, alpha)
            y_pred_function = selector.predict_lasso_regression

        aic_score = selector.aic(X_poly, y, beta, model)

        # Print results
        print(f"\nModel: {model.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {k_fold_mse:.4f}")
        print(f"Bootstrapping MSE: {bootstrap_mse:.4f}")
        print(f"AIC Score: {aic_score:.4f}")

        results[model] = {
            "k_fold_mse": k_fold_mse,
            "bootstrap_mse": bootstrap_mse,
            "aic_score": aic_score
        }

    # Identify and print the best model
    selector.best_model(results)
    selector.visualize_model_comparison(results, dataset_name)

if __name__ == "__main__":
    test_wine_quality()
    test_polynomial_regression()
    test_boston_housing()
    test_concrete_strength()
    test_diabetes()