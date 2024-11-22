import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from models.gradient_reg import GradientBoostingRegressor  

# Generate Random Regression Dataset
def generate_random_data(n_samples=1000, n_features=5, noise=0.1, random_state=42):
    """
    Generates a random regression dataset.
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features)
    coefficients = np.random.rand(n_features)
    y = X @ coefficients + noise * np.random.randn(n_samples)
    return X, y

def main():
    # Generate the random dataset
    X, y = generate_random_data(n_samples=500, n_features=5, noise=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)

    print("Training the Gradient Boosting Regressor...")
    gbr.fit(X_train, y_train)

    y_pred = gbr.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()