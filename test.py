
from GradientBoost import *
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Generate data
    np.random.seed(42) # For reproducibility
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    y = X[:, 0]**2 + 2*X[:, 1] + np.random.normal(0, 0.1, 100)  # Nonlinear target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = GradientBoosting(n_trees=200, lr=0.05, max_depth=4)
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred)**2)
    print(f"Test MSE: {mse:.4f}")