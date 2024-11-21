# Import necessary libraries
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# GradientBoosting class is in a file named gradient_boosting.py
from gradient_boosting import GradientBoosting

def main():
    # Load the California housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the testing data with the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Initialize the GradientBoosting model
    model = GradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)

    # Train the model on the scaled training data
    model.fit(X_train_scaled, y_train)

    # Predict the scaled test set
    predictions = model.predict(X_test_scaled)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error on Test Set:", mse)

if __name__ == "__main__":
    main()
