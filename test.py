# test.py
import numpy as np
import pandas as pd
from model import LinearRegression, RidgeRegression, ModelSelector
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path, delimiter=';') #Be sure the dataset used is separeted by ',' or ';' , and change the delimiter accordingly
    #df = pd.read_csv(file_path)
    df = df.dropna() # Removing entire rows where there is any null in any column
    # Separate features and target
    X = df.drop('medv', axis=1).values
    y = df['medv'].values
    
    # Standardize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y


def split_data(X, y, test_size=0.2):
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    test_size = int(test_size * n_samples)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return (X[train_indices], X[test_indices], 
            y[train_indices], y[test_indices])


def plot_results(results):
    models = list(results.keys())
    metrics = ['test_mse', 'kfold_mse', 'bootstrap_mse']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot MSE comparisons
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax1.bar(x + i*width, values, width, label=metric)
    
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels(models)
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE Comparison')
    ax1.legend()
    
    # Plot AIC scores
    aic_scores = [results[model]['aic'] for model in models]
    ax2.bar(models, aic_scores)
    ax2.set_ylabel('AIC Score')
    ax2.set_title('AIC Comparison')
    
    plt.tight_layout()
    plt.show()


def main():
    # Load and prepare data
    X, y = load_and_preprocess_data('Boston.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)

    
    # Initialize models
    models = {
        'linear': LinearRegression(),
        'ridge': RidgeRegression(alpha=1.0)
    }
    
    # Initialize model selector
    selector = ModelSelector(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        results[name] = selector.evaluate_model(model)
        
        print(f"\n{name.upper()} REGRESSION RESULTS:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['test_mse'])
    print(f"\nBest model: {best_model[0]} (Test MSE: {best_model[1]['test_mse']:.4f})")
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()