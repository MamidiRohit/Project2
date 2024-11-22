import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from models.gradient_reg import GradientBoostingRegressor  # Replace `your_file_name` with the file name containing your implementation

# Load the dataset
file_path = "tests/laptop_prices.csv"  # Replace with the actual path to your dataset
data = pd.read_csv(file_path)

# Select features and target
features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage']
target = 'Price_euros'

X = data[features].values
y = data[target].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)

# Train the model
print("Training the Gradient Boosting Regressor...")
gbr.fit(X_train, y_train)

# Make predictions
y_pred = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error (MSE): {mse:.2f}")
print(f"Test R² Score: {r2:.4f}")