import numpy as np
from models.gradient_boosting_tree import GradientBoostingTree
from utils.evaluation import evaluate_model

def generate_synthetic_data(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    np.random.seed(random_state)
    # Generate random feature data (X)
    X = np.random.randn(n_samples, n_features)
    # Generate true coefficients for the linear model (random weights)
    true_coefficients = np.random.randn(n_features)
    # Generate the target variable (y) as a linear combination of features plus noise
    y = np.dot(X, true_coefficients) + noise * np.random.randn(n_samples)

    return X, y

def test_gradient_boosting_tree():
    # Generate synthetic regression data
    X, y = generate_synthetic_data(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    # Split data into training and test sets (80-20 split)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Define hyperparameters for the test
    learning_rate = 0.05
    n_estimators = 200
    max_depth = 3
    min_samples_split = 10
    
    # Initialize the Gradient Boosting Tree model
    model = GradientBoostingTree(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test data
    predictions = model.predict(X_test)
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    # Print the results
    print("Test MSE:", mse)
    print("Test R²:", r2)
    print("Predictions on Test Set:")
    print(predictions[:10])

# Run the test
if __name__ == "__main__":
    test_gradient_boosting_tree()
