import csv
import random
import math
import matplotlib.pyplot as plt

def load_csv(file_path, target_column):
    """
    Load a CSV file and split into features (X) and target (y).
    
    Parameters:
        file_path: Path to the CSV file.
        target_column: Name of the target column.

    Returns:
        X: List of feature rows (2D list).
        y: List of target values (1D list).
    """
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    
    X = []
    y = []
    for row in data:
        y.append(float(row[target_column]))
        X.append([float(value) for key, value in row.items() if key != target_column])
    
    return X, y

def mean_squared_error(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_residual = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1 - (ss_residual / ss_total)

def split_data(X, y, indices):
    X_split = [X[i] for i in indices]
    y_split = [y[i] for i in indices]
    return X_split, y_split

def k_fold_cv(model, X, y, k=5, metric='mse', random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    indices = list(range(len(X)))
    random.shuffle(indices)
    fold_size = len(X) // k
    scores = []

    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = indices[:i * fold_size] + indices[(i + 1) * fold_size:]
        
        X_train, y_train = split_data(X, y, train_indices)
        X_test, y_test = split_data(X, y, test_indices)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if metric == 'mse':
            scores.append(mean_squared_error(y_test, y_pred))
        elif metric == 'r2':
            scores.append(r_squared(y_test, y_pred))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    return sum(scores) / len(scores)

def bootstrap(model, X, y, num_samples=100, metric='mse', random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    scores = []
    n = len(X)

    for _ in range(num_samples):
        bootstrap_indices = [random.randint(0, n - 1) for _ in range(n)]
        oob_indices = list(set(range(n)) - set(bootstrap_indices))
        
        X_sample, y_sample = split_data(X, y, bootstrap_indices)
        X_oob, y_oob = split_data(X, y, oob_indices)
        
        if not oob_indices:
            continue

        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_oob)
        
        if metric == 'mse':
            scores.append(mean_squared_error(y_oob, y_pred))
        elif metric == 'r2':
            scores.append(r_squared(y_oob, y_pred))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    return sum(scores) / len(scores)

def calculate_aic(model, X, y):
    model.fit(X, y)
    y_pred = model.predict(X)
    resid = [yt - yp for yt, yp in zip(y, y_pred)]
    n = len(y)
    p = len(X[0])
    rss = sum(r ** 2 for r in resid)
    return n * math.log(rss / n) + 2 * p

def plot_results(y_true, y_pred):
    """
    Plot observed vs. predicted values and residuals.

    Parameters:
        y_true: List of true target values.
        y_pred: List of predicted target values.
    """
    # Observed vs. Predicted
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit")
    plt.title("Observed vs Predicted")
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)

    # Residuals
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k')
    plt.axhline(y=0, color='r', linestyle='--', label="Zero Residual Line")
    plt.title("Residuals")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(True)

    # Show plots
    plt.tight_layout()
    plt.show()

class LinearRegression:
    def __init__(self):
        self.coef_ = []
        self.intercept_ = 0

    def fit(self, X, y):
        n = len(X)
        p = len(X[0])
        X_flat = [x + [1] for x in X]
        XtX = [[sum(X_flat[i][k] * X_flat[i][j] for i in range(n)) for j in range(p + 1)] for k in range(p + 1)]
        Xty = [sum(X_flat[i][j] * y[i] for i in range(n)) for j in range(p + 1)]
        self.coef_, self.intercept_ = self.solve_linear_system(XtX, Xty)

    def predict(self, X):
        return [sum(c * x for c, x in zip(self.coef_, xi)) + self.intercept_ for xi in X]

    def solve_linear_system(self, A, b):
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
                b[j] -= factor * b[i]
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
        return x[:-1], x[-1]

if __name__ == "__main__":
    # Example CSV Usage
    file_path = "mlp\small_test.csv"  # Replace with your CSV file path
    target_column = "target"  # Replace with your target column name

    # Load data from CSV
    X, y = load_csv("mlp\small_test.csv", "y")

    # Create a Linear Regression model
    model = LinearRegression()

    # K-Fold Cross-Validation
    kfold_score = k_fold_cv(model, X, y, k=5, metric='mse', random_seed=42)
    print(f"K-Fold Cross-Validation MSE: {kfold_score:.4f}")

    # Bootstrapping
    bootstrap_score = bootstrap(model, X, y, num_samples=100, metric='mse', random_seed=42)
    print(f"Bootstrapping MSE: {bootstrap_score:.4f}")

    # AIC
    aic_score = calculate_aic(model, X, y)
    print(f"AIC: {aic_score:.4f}")
    
    # Fit the model and get predictions
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot results
    plot_results(y, y_pred)
