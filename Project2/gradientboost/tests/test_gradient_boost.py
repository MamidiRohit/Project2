import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import sys
import os

# Insert the project root directory into the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gradientboost.models.GradientBoost import GradientBoostModel  

def load_csv_data(file_path):
    """
    Load real-world data from a CSV file and handle missing values and categorical features.
    """
    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("The CSV file is empty.")
    
    # Drop columns that are completely empty
    data = data.dropna(axis=1, how="all")

    # Handle missing values
    for col in data.columns:
        if data[col].dtype == "object":
            # Fill missing values for categorical columns with the mode
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # Convert numerical columns to numeric, coercing errors to NaN, then fill with the mean
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(data[col].mean())

    # Ensure no missing values remain
    if data.isnull().any().any():
        raise ValueError("Data contains unresolved missing values after processing.")

    # Separate features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Handle categorical target values using Label Encoding for classification tasks
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Convert target column to numeric
    y = np.array(y, dtype=np.float64)

    # Check for NaN values in NumPy array
    if np.isnan(y).any():
        raise ValueError("Target column contains invalid values (NaN) after processing.")

    # Apply one-hot encoding for categorical columns
    if X.select_dtypes(include=["object"]).shape[1] > 0:
        X = pd.get_dummies(X, drop_first=True)

    return X.values, y

def preprocess_data(X, y):
    """
    Preprocess the features and target for better performance.
    """
    # Ensure X is numeric
    X = np.array(X, dtype=np.float64)

    # Add noise to zero variance features
    zero_variance_mask = np.var(X, axis=0) <= 1e-6
    if np.any(zero_variance_mask):
        print(f"Warning: Zero variance detected. Adding noise to {np.sum(zero_variance_mask)} features.")
        noise = np.random.normal(0, 1e-6, X[:, zero_variance_mask].shape)
        X[:, zero_variance_mask] += noise

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def evaluate_model(y_true, y_pred):
    """
    Automatically detect task type (classification or regression) and evaluate the model's performance.
    """
    if np.issubdtype(y_true.dtype, np.integer):  # Classification task
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Is Accuracy Good?: {'Yes' if accuracy >= 0.7 else 'No'}")
    else:  # Regression task
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        std_y = np.std(y_true)
        threshold = std_y ** 2
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print(f"Standard Deviation of Target (y): {std_y:.2f}")
        print(f"Threshold for Good MSE (Variance of y): {threshold:.2f}")
        print(f"Is MSE Good?: {'Yes' if mse < threshold else 'No'}")

def test_real_world_data(file_path):
    """
    Test Gradient Boosting on real-world data from a CSV file.
    """
    X, y = load_csv_data(file_path)
    X, y = preprocess_data(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Gradient Boosting model
    model = GradientBoostModel(n_estimators=200, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nReal-World Data Evaluation:")
    evaluate_model(y_test, y_pred)

    print("Sample Predictions:", y_pred[:10])
    print("Sample True Values:", y_test[:10])

def test_pathological_cases():
    """
    Test Gradient Boosting on pathological cases with added random noise for better test coverage.
    Handles both classification and regression tasks based on data type.
    """
    cases = [
        ("High Dimensionality", np.random.randn(10, 100), np.random.randint(0, 2, 10)),  # Classification
        ("Zero Variance Feature", 
         np.array([[1, 1, 1]] * 4) + np.random.normal(0, 1e-6, (4, 3)), 
         np.array([0, 1, 0, 1])),  # Classification
        ("Perfectly Collinear Features", 
         np.array([[1, 2, 4], [1, 2, 4], [2, 4, 8], [2, 4, 8]]) + np.random.normal(0, 1e-6, (4, 3)), 
         np.array([0, 1, 0, 1])),  # Classification
        ("Extremely Large Values", np.random.randn(100, 10) * 1e10, np.random.randn(100)),  # Regression
        ("Extremely Small Values", np.random.randn(100, 10) * 1e-10, np.random.randn(100)),  # Regression
    ]

    for case_name, X, y in cases:
        print(f"\nTesting {case_name}...")
        try:
            # Detect task type: classification or regression
            is_classification = np.issubdtype(y.dtype, np.integer)
            
            model = GradientBoostModel(n_estimators=50, learning_rate=0.1, max_depth=3)
            model.fit(X, y)
            y_pred = model.predict(X)

            # Evaluate based on task type
            if is_classification:
                accuracy = accuracy_score(y, np.round(y_pred))
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Is Accuracy Good?: {'Yes' if accuracy >= 0.7 else 'No'}")
            else:
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                std_y = np.std(y)
                threshold = std_y ** 2
                print(f"Mean Squared Error (MSE): {mse:.2f}")
                print(f"R² Score: {r2:.2f}")
                print(f"Standard Deviation of Target (y): {std_y:.2f}")
                print(f"Threshold for Good MSE (Variance of y): {threshold:.2f}")
                print(f"Is MSE Good?: {'Yes' if mse < threshold else 'No'}")
            
            # Display sample predictions
            print("Sample Predictions:", y_pred[:10])
            print("Sample True Values:", y[:10])
        except Exception as e:
            print(f"{case_name} failed: {e}")


# Run the tests
if __name__ == "__main__":
    try:
        test_real_world_data("seattle_weather.csv")  # Replace with the file path
    except Exception as e:
        print(f"Error during real-world data testing: {e}")

    test_pathological_cases()
