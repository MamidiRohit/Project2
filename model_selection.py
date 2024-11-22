import numpy as np
from typing import Callable, Dict, Any, Tuple

try:
    import matplotlib.pyplot as plt
except Exception as e:
    ...

def linear_regression(X:np.ndarray, y:np.ndarray) -> np.ndarray:
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta

def ridge_regression(X:np.ndarray, y:np.ndarray, alpha:float=1.0):
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    I = np.eye(X_b.shape[1])
    I[0, 0] = 0
    beta = np.linalg.inv(X_b.T @ X_b + alpha * I) @ X_b.T @ y
    return beta

def k_fold_cross_validation(X:np.ndarray, y:np.ndarray, k:int=5, model_fn:Callable=linear_regression, **model_params) -> float:
    n = X.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)
    mse_scores = np.zeros(k)
    
    for i in range(k):
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        beta = model_fn(X_train, y_train, **model_params)
        X_test_b = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        predictions = X_test_b @ beta
        mse_scores[i] = np.mean((y_test - predictions) ** 2)
    
    return np.mean(mse_scores)

def bootstrapping(X:np.ndarray, y:np.ndarray, B:int=1000, model_fn=linear_regression, **model_params):
    n = X.shape[0]
    mse_scores = np.zeros(B)
    
    for b in range(B):
        indices = np.random.choice(n, n, replace=True)
        X_b = X[indices]
        y_b = y[indices]
        
        beta = model_fn(X_b, y_b, **model_params)
        X_test_b = np.hstack((np.ones((n, 1)), X))
        predictions = X_test_b @ beta
        mse_scores[b] = np.mean((y - predictions) ** 2)
    
    return np.mean(mse_scores)

def aic(X:np.ndarray, y:np.ndarray, beta) -> Tuple[float, float]:
    n = X.shape[0]
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = X_b @ beta
    ssr = np.sum((y - y_pred) ** 2)
    p = len(beta) - 1
    aic_value = n * np.log(ssr / n) + 2 * (p + 1)
    return aic_value, ssr

def evaluate_models(models:Dict[str,any], X:np.ndarray, y:np.ndarray, k:int=5, B:int=1000):
    results = {}
    for name, model_info in models.items():
        model_fn = model_info['fn']
        model_params = model_info.get('params', {})
        beta = model_fn(X, y, **model_params)
        cv_mse = k_fold_cross_validation(X, y, k, model_fn, **model_params)
        bootstrap_mse = bootstrapping(X, y, B, model_fn, **model_params)
        aic_val, ssr = aic(X, y, beta)
        results[name] = {
            "Coefficients": beta,
            "K-Fold MSE": cv_mse,
            "Bootstrapping MSE": bootstrap_mse,
            "AIC": aic_val,
            "SSR": ssr,
        }
    return results

def verify_coefficients(X:np.ndarray, y:np.ndarray, beta:np.ndarray, model_fn:callable, **model_params) -> bool:
    beta_sanity = model_fn(X, y, **model_params)
    return np.allclose(beta, beta_sanity)

def verify_aic(X:np.ndarray, y:np.ndarray, beta:np.ndarray, aic_value:float) -> bool:
    n = X.shape[0]
    X_b = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = X_b @ beta
    ssr = np.sum((y - y_pred) ** 2)
    p = len(beta) - 1
    aic_manual = n * np.log(ssr / n) + 2 * (p + 1)
    return np.isclose(aic_value, aic_manual)


np.random.seed(42)
n = 100
p = 1
X = np.random.rand(n, p)
beta_true = np.array([1])
y = X @ beta_true + np.random.randn(n)

models = {
    "Linear Regression": {"fn": linear_regression},
    "Ridge Regression": {"fn": ridge_regression, "params": {"alpha": 1.0}},
}

results = evaluate_models(models, X, y, k=5, B=1000)

if plt:
    def visualize(evaluate_models, X, y, models, results):
        kfold_mse = [result['K-Fold MSE'] for result in results.values()]
        bootstrap_mse = [result['Bootstrapping MSE'] for result in results.values()]
        aic_values = [result['AIC'] for result in results.values()]

        results = evaluate_models(models, X, y, k=5, B=1000)
        kfold_mse = [result['K-Fold MSE'] for result in results.values()]
        bootstrap_mse = [result['Bootstrapping MSE'] for result in results.values()]
        aic_values = [result['AIC'] for result in results.values()]
        models_names = list(results.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        bar_width = 0.35
        index = np.arange(len(models_names))
        bars1 = ax1.bar(index, kfold_mse, bar_width, label='K-Fold MSE', color='blue')
        bars2 = ax1.bar(index + bar_width, bootstrap_mse, bar_width, label='Bootstrapping MSE', color='green')

        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels(models_names, rotation=45)

        ax1.set_title('Model Evaluation: MSE Comparison', fontsize=14)
        def autolabel(bars, ax):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        autolabel(bars1, ax1)
        autolabel(bars2, ax1)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=10)
        ax1.set_yticks([])
        bars3 = ax2.bar(index, aic_values, bar_width * 2, label='AIC', color='red')
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_xticks(index)
        ax2.set_xticklabels(models_names, rotation=45)
        ax2.set_title('Model Evaluation: AIC Comparison', fontsize=14)

        autolabel(bars3, ax2)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=10)
        ax2.set_yticks([])
        plt.tight_layout()
        plt.show()
        return results

    results = visualize(evaluate_models, X, y, models, results)


for name, result in results.items():
    print(f"Model: {name}")
    print(f"Coefficients: {result['Coefficients']}")
    print(f"K-Fold MSE: {result['K-Fold MSE']}")
    print(f"Bootstrapping MSE: {result['Bootstrapping MSE']}")
    print(f"AIC: {result['AIC']}")
    print(f"SSR: {result['SSR']}")
    print(f"Verification (Coefficients): {verify_coefficients(X, y, result['Coefficients'], models[name]['fn'], **models[name].get('params', {}))}")
    print(f"Verification (AIC): {verify_aic(X, y, result['Coefficients'], result['AIC'])}")
    print("\n")
    print("-"*30)