import numpy as np
import pandas as pd

class SyntheticDataGenerator:
    def __init__(self, rows, cols, noise_level=0.4, random_seed=20):
        self.rows = rows
        self.cols = cols
        self.noise_level = noise_level
        self.random_seed = random_seed

    def generate_data(self):
        np.random.seed(self.random_seed)
        X = np.random.randn(self.rows, self.cols)
        coefficients = np.random.randn(self.cols)
        coefficients[2:5] = 0  # Set some coefficients to 0 for sparsity
        noise = np.random.randn(self.rows) * self.noise_level
        y = X @ coefficients + noise

        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(self.cols)])
        df['target'] = y

        return df.drop(columns='target').to_numpy(), df['target'].to_numpy()


class CustomDataGenerator:
    def __init__(self, coefficients, num_samples, intercept=0, value_range=(-10, 10), noise_scale=1, random_seed=86413459):
        self.coefficients = np.array(coefficients)
        self.num_samples = num_samples
        self.intercept = intercept
        self.value_range = value_range
        self.noise_scale = noise_scale
        self.random_seed = random_seed

    def generate_linear_data(self):
        np.random.seed(self.random_seed)
        rng = np.random
        X = rng.uniform(low=self.value_range[0], high=self.value_range[1], size=(self.num_samples, len(self.coefficients)))
        y = X @ self.coefficients + self.intercept
        noise = rng.normal(loc=0.0, scale=self.noise_scale, size=y.shape)
        return X, y + noise
