# Project 2
## Group Name: GRADIENT SQUADS
## Group Members
- Laasya Priya Vemuri (CWID: A20561469)
- Dyuti Dasary (CWID: A20546872)
- Charan Reddy Kandula Venkata (CWID: A20550020)
- Niranjaan Veeraragahavan Munuswamy (CWID: A20552057)

# Gradient Boost Model Implementation
### 1. What Does the Model You Have Implemented Do and When Should It Be Used?

The Gradient Boost Model (GBM) is a custom implementation of a powerful machine learning algorithm based on the gradient boosting technique. Gradient boosting builds an ensemble of decision trees, where each subsequent tree corrects the errors of its predecessor, optimizing a given loss function.

**Key Features of the Model**
**Gradient-Based Optimization:** Minimizes the loss function iteratively by learning from gradients of residuals (errors).
**Flexible Use Cases:** Handles both classification and regression tasks effectively.
**Automatic Preprocessing:** Includes mechanisms to handle missing values, encode categorical variables, and standardize numerical features for better model performance.

#### When Should It Be Used?
The GBM model is highly versatile and should be used in scenarios such as:

**Regression Tasks:** Predicting continuous values like house prices, sales forecasting, and temperature modeling.
**Classification Tasks:** Categorizing items like detecting spam emails or classifying animal breeds.
**Handling Non-Linear Data:** Addresses complex, non-linear relationships in the dataset.
**Medium-Sized Datasets:** Suitable for datasets where computational overhead is manageable with proper parameter tuning.
**Noise-Resilient Use Cases:** Learns robust patterns from noisy data due to its iterative boosting process.

### 2. How Did You Test Your Model to Determine if It Is Working Reasonably Correctly?

**Testing Strategies**
The model was rigorously tested on both real-world and synthetic datasets under various conditions to ensure robustness.

**a. Real-World Data Testing**
Datasets Used:

- seattle_weather.csv: Weather data for regression tasks.
- dog_breeds.csv: Classification dataset for breed identification.
- laptop_prices.csv: Regression dataset for predicting laptop prices.

**Testing Steps:**

**-Data Preprocessing:** Missing values were handled through imputation, categorical features were encoded, and numerical features were standardized.
**-Train-Test Split:** Split datasets into 70% training and 30% testing data.


**Performance Metrics:**
**-Regression Tasks:** Evaluated using Mean Squared Error (MSE) and R² Score.
**-Classification Tasks:** Evaluated using Accuracy.

**b. Pathological Test Cases**
Synthetic datasets were designed to stress-test the model’s robustness under extreme conditions:

**High Dimensionality:**
- Dataset with more features than samples.
- Verified performance on sparse data.
**Zero Variance Features:**
- Tested the model’s ability to handle features with constant values.
- Noise was added to ensure stability.
**Perfectly Collinear Features:**
- Evaluated robustness against multicollinearity (features that are linear combinations of others).
**Extreme Scaling:**
- Datasets with very large (e.g., 1e10) and very small (e.g., 1e-10) values.
- Ensured numerical stability through preprocessing.
**Skewed Target Values:**
- Examined the model’s handling of highly skewed distributions.


### 3. What Parameters Have You Exposed to Users of Your Implementation in Order to Tune Performance?

The model provides several customizable parameters to control training and optimize performance:

**Model Parameters**
**n_estimators:** Number of boosting iterations (trees).
- Default: 100.
- Usage: Increase to improve accuracy; may increase training time.

**learning_rate:** Shrinkage parameter that scales the contribution of each tree.
- Default: 0.1.
- Usage: Smaller values require more trees but improve generalization.

**max_depth:** Maximum depth of individual trees.
- Default: 3.
- Usage: Controls complexity; deeper trees capture more detail but risk overfitting.

**Basic Methods**
**fit(X, y):**
- Trains the model on feature matrix X and target vector y.  
**predict(X):**
- Predicts outcomes for unseen data X.
- Outputs continuous values for regression and probabilities for classification.

**Advanced Features:**
**Automatic Preprocessing:**
- Handles missing values by imputing with mean or mode.
- Encodes categorical features using one-hot encoding.
- Standardizes numerical features to improve gradient updates.

## Basic Usage Examples

### Example 1: Synthetic Data Example
This example demonstrates how to train and make predictions using synthetic data. You can run this code directly without any changes:

```python
import sys
import os
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Insert the project root directory into the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gradientboost.models.GradientBoost import GradientBoostModel

# Generate synthetic data for testing
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
gbr = GradientBoostModel(n_estimators=200, learning_rate=0.1, max_depth=3)
gbr.fit(X_train, y_train)

# Predictions on the test set
predictions = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
std_y = np.std(y_test)  # Standard deviation of the target variable

# Determine if MSE is "good"
threshold = std_y**2  # Variance of the target
is_mse_good = mse < threshold

# Display results
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
print("Standard Deviation of Target (y):", std_y)
print("Threshold for Good MSE (Variance of y):", threshold)
print("Is MSE Good?:", "Yes" if is_mse_good else "No")
print("Sample Predictions:", predictions[:10])
print("Sample True Values:", y_test[:10])

```

### Example 2: Using Your Own CSV File and Pathological Cases
To use the model with your own data:
- Replace "dog_breeds.csv" with the name of your CSV file.
- Ensure the file is placed in the correct directory (gradientboost/tests).
- Ensure your CSV file contains a header row and the last column should be the target variable (y), and all other columns will be treated as features (X).

```python
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
        test_real_world_data("dog_breeds.csv")  # Replace with the file path
    except Exception as e:
        print(f"Error during real-world data testing: {e}")

    test_pathological_cases()
```

### Running tests with Pytest
Once you've modified the file path, you can use pytest to run all tests, including real-world data tests and pathological case tests.
- Open a terminal in the directory containing your test script.
- Run Pytest with the following command:
pytest test_gradient_boost.py

This will execute all tests in test_gradient_boost.py, which includes:
- **Real-World Data Testing:** Validates model performance on the CSV file you specified.
- **Pathological Cases:** Tests the model’s resilience under extreme conditions like high dimensionality, collinear features, and extreme values.

### Understanding the Test Outputs
**For Real-World Data:** Look for output such as Mean Absolute Error (for regression) to evaluate model performance.
**For Pathological Cases:** Outputs will indicate whether the model successfully handled each challenging case or encountered issues. Any assertion failures will provide feedback on specific issues with predictions.

### 4. Are There Specific Inputs That Your Implementation Has Trouble With?

**Challenges Faced**

**-Imbalanced Datasets:**

- Issue: The model struggled with datasets where the target variable was imbalanced (e.g., most values were concentrated in one range).
- Workaround: Sample weighting or resampling techniques were considered to balance the dataset.

**-High Dimensionality:**

- Issue: Training time increased significantly when the number of features exceeded the number of samples.
- Workaround: Dimensionality reduction techniques, such as Principal Component Analysis (PCA), were identified as potential solutions to reduce feature space.

**-Sparse Datasets:**

- Issue: The model did not perform optimally on sparse data with many zero values.
- Workaround: The implementation could be modified to handle sparse matrix formats more efficiently.

## Future Improvements
Given more time, the following enhancements would have been implemented:

**-Classification Support:** The model would have been extended to handle multi-class classification with appropriate loss functions.
**-Early Stopping:**- Mechanisms would have been introduced to halt training when validation performance stopped improving.
**-Feature Importance:**- Methods to calculate and visualize feature importance for better interpretability would have been developed.
**-Adaptive Learning Rate:**- Dynamic learning rate adjustment for better convergence would have been implemented.
**-Built-in Cross-Validation:**- Automatic hyperparameter tuning via grid search or random search would have been added.

