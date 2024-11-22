import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model_Selector:
    def AIC_calculator(self, X, y, beta):
    
        #Calculating the AIC scores for the models
    
        length_y = len(y)
        preds = X @ beta
        residual = y - preds
        rss = np.sum(residual**2)
        square_of_sigma = rss / length_y 
        log_likelihood = -0.5 * length_y * (np.log(2 * np.pi * square_of_sigma) + 1)
        
        length_beta = len(beta)  # Number of coefficients
        aic_scores = 2 * length_beta - 2 * log_likelihood
        return aic_scores

    def linear_regression_model(self, X, y):

        #Implement Linear regression model

        XTX = X.T @ X
        XTy = X.T @ y
        beta = np.linalg.solve(XTX, XTy)
        return beta

    def ridge_regression_model(self, X, y, alpha=1.0):
        #Implement Ridge Regression model.
        
        features_n = X.shape[1]
        XTX = X.T @ X + alpha * np.eye(features_n)
        XTy = X.T @ y
        beta = np.linalg.solve(XTX, XTy)
        return beta

    def lasso_regression_model(self, X, y, alpha=1.0, max_iter=1000, tol=1e-4):
        #Implement Lasso Regression model

        n, m = X.shape
        beta = np.zeros(m)
        for _ in range(max_iter):
            beta_value = beta.copy()
            for j in range(m):
                residual = y - X @ beta + beta[j] * X[:, j]
                rho = X[:, j].T @ residual
                if rho < -alpha:
                    beta[j] = (rho + alpha) / (X[:, j] @ X[:, j])
                elif rho > alpha:
                    beta[j] = (rho - alpha) / (X[:, j] @ X[:, j])
                else:
                    beta[j] = 0
            if np.linalg.norm(beta - beta_value, ord=1) < tol:
                break
        return beta

    def k_fold_cross_validation(self, X, y, model_fn, alpha=None, k=5):
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k
        aic_scores = []

        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            if alpha is not None:
                beta = model_fn(X_train, y_train, alpha)
            else:
                beta = model_fn(X_train, y_train)

            aic_score = self.AIC_calculator(X_test, y_test, beta)
            aic_scores.append(aic_score)

        return aic_scores

    def bootstrap_validation(self, X, y, model_fn, alpha=None, n_bootstraps=100):
        n = len(y)
        aic_scores = []

        for _ in range(n_bootstraps):
            bootstrap_indices = np.random.choice(n, size=n, replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

            if alpha is not None:
                beta = model_fn(X_bootstrap, y_bootstrap, alpha)
            else:
                beta = model_fn(X_bootstrap, y_bootstrap)

            aic_score = self.AIC_calculator(X_bootstrap, y_bootstrap, beta)
            aic_scores.append(aic_score)

        return aic_scores

def summarize_results(aic_scores):
    n = len(aic_scores)
    mean_aic = sum(aic_scores) / n
    variance = sum((x - mean_aic) ** 2 for x in aic_scores) / n
    std_dev = variance ** 0.5
    return mean_aic, std_dev

def plot_results(results):
    """
    Plot the AIC results for k-Fold and Bootstrapping across models using horizontal bars.
    """
    models = list(results.keys())
    mean_aic_kf = [results[model]["mean_aic_kf"] for model in models]
    mean_aic_bootstrap = [results[model]["mean_aic_bootstrap"] for model in models]
    kf_std_dev = [results[model]["kf_std_dev"] for model in models]
    bootstrap_std_dev = [results[model]["bootstrap_std_dev"] for model in models]
    
    # k-Fold Horizontal Bar Plot with Error Bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(models, mean_aic_kf, color='green', xerr=kf_std_dev, capsize=5)
    ax.set_title('k-Fold Cross-Validation AIC')
    ax.set_ylabel('Models')
    ax.set_xlabel('AIC')
    plt.show()

    # Bootstrap AIC Horizontal Bar Plot with Error Bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(models, mean_aic_bootstrap, color='purple', xerr=bootstrap_std_dev, capsize=5)
    ax.set_title('Bootstrap AIC')
    ax.set_ylabel('Models')
    ax.set_xlabel('AIC')
    plt.show()

    # Boxplots for AIC Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results[model]["kf_scores"] for model in models], vert=True)
    ax.set_title('k-Fold AIC Distribution')
    ax.set_xticklabels(models)
    ax.set_ylabel('AIC')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results[model]["bootstrap_scores"] for model in models], vert=True)
    ax.set_title('Bootstrap AIC Distribution')
    ax.set_xticklabels(models)
    ax.set_ylabel('AIC')
    plt.show()

def generic_process(data, X_columns, y_column):
    """
    Generic process to evaluate AIC for Linear, Ridge, and Lasso models on any dataset.
    """
    # Split features and target
    X = data[X_columns].values
    y = data[y_column].values

    # Initialize Model_Selector
    selector = Model_Selector()

    # Define models
    models = {
        'linear': selector.linear_regression_model,
        'ridge': lambda X, y: selector.ridge_regression_model(X, y, alpha=1.0),
        'lasso': lambda X, y: selector.lasso_regression_model(X, y, alpha=1.0)
    }

    results = {}

    for model_name, model_fn in models.items():
        print(f"Evaluating model: {model_name.capitalize()}")

        # Perform k-Fold Cross-Validation
        aic_scores_kf = selector.k_fold_cross_validation(X, y, model_fn, k=5)

        # Perform Bootstrapping
        aic_scores_bootstrap = selector.bootstrap_validation(X, y, model_fn, n_bootstraps=100)

        # Calculate mean and standard deviation
        mean_aic_kf, std_aic_kf = summarize_results(aic_scores_kf)
        mean_aic_bootstrap, std_aic_bootstrap = summarize_results(aic_scores_bootstrap)

        # Save the results
        results[model_name] = {
            "mean_aic_kf": mean_aic_kf,
            "mean_aic_bootstrap": mean_aic_bootstrap,
            "kf_std_dev": std_aic_kf,
            "bootstrap_std_dev": std_aic_bootstrap,
            "kf_scores": aic_scores_kf,
            "bootstrap_scores": aic_scores_bootstrap
        }

    # Find the best model
    best_model = None
    best_mean_aic = float('inf')

    for model_name, scores in results.items():
        avg_aic = (scores["mean_aic_kf"] + scores["mean_aic_bootstrap"]) / 2
        print(f"\n{model_name.capitalize()} - Mean AIC: k-Fold: {scores['mean_aic_kf']:.3f}, Bootstrapping: {scores['mean_aic_bootstrap']:.3f}")

        if avg_aic < best_mean_aic:
            best_mean_aic = avg_aic
            best_model = model_name

    print(f"\nBest Model: {best_model.capitalize()} with an average AIC of {best_mean_aic:.3f}")

    # Plot results
    plot_results(results)


# Example Usage:
if __name__ == "__main__":

    data = pd.read_csv('patient_data.csv')

    feature_columns = ['RR_Interval','QRS_Duration','QT_Interval']
    target_column = 'Heart_Rate'

    # Run the process
    generic_process(data, feature_columns, target_column)