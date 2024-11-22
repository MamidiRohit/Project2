import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples and features
n_samples = 1000
n_features = 5

# Generate a random base feature
base_feature = np.random.normal(0, 1, n_samples)

# Create a highly correlated dataset
correlated_data = np.array([base_feature + np.random.normal(0, 0.1, n_samples) for _ in range(n_features)]).T

# Convert to a DataFrame
columns = [f"Feature_{i+1}" for i in range(n_features)]
df = pd.DataFrame(correlated_data, columns=columns)

# Add a target variable with a high correlation to one of the features
df['Target'] = df['Feature_1'] * 2 + np.random.normal(0, 0.1, n_samples)

# Save to CSV
output_file = "highly_correlated_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")
