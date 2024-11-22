import numpy as np
import pandas as pd
# Adjust the correlation matrix for partial correlation
partial_correlation_matrix = np.array([
    [1.0, 0.3, 0.2, 0.1, 0.0],
    [0.3, 1.0, 0.4, 0.2, 0.1],
    [0.2, 0.4, 1.0, 0.3, 0.2],
    [0.1, 0.2, 0.3, 1.0, 0.4],
    [0.0, 0.1, 0.2, 0.4, 1.0]
])
mean = [0, 0, 0, 0, 0]
# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate a multivariate normal distribution for partially correlated predictors
X_partial = np.random.multivariate_normal(mean, partial_correlation_matrix, size=n_samples)

# Create a DataFrame for predictors
df_partial = pd.DataFrame(X_partial, columns=['X1', 'X2', 'X3', 'X4', 'X5'])

# Generate the target variable y with a weaker non-linear relationship and noise
noise_partial = np.random.normal(0, 0.5, n_samples)
df_partial['y'] = (
    1.5 * df_partial['X1'] +
    0.5 * np.log(np.abs(df_partial['X2']) + 1) -
    0.8 * df_partial['X3'] +
    0.2 * df_partial['X4']**2 +
    noise_partial
)

# Save to CSV
file_path_partial = "data/partially_correlated_dataset.csv"
df_partial.to_csv(file_path_partial, index=False)
