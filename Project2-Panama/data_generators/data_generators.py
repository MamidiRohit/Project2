import numpy as np
from scipy.signal import sawtooth

# Test 1: Linear data with one feature
def linear_data_generator1(m, b, range_, N, seed):
    """
    Generates 1D linear data with noise.
    Args:
        m: Slope of the linear function.
        b: Intercept of the linear function.
        range_: Tuple (low, high) for the range of X values.
        N: Number of samples.
        seed: Random seed for reproducibility.
    Returns:Tuple (X, y) where X is the feature array, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=N)
    ys = m * sample + b
    noise = rng.normal(loc=0.0, scale=3.0, size=N)
    return sample.reshape(-1, 1), ys + noise


# Test 2: Linear data with multiple features
def linear_data_generator2(m, b, range_, N, seed):
    """
    Generates linear data with multiple features and noise.
    Args:
        m: Array of coefficients for the linear combination.
        b: Intercept of the linear function.
        range_: Tuple (low, high) for the range of X values.
        N: Number of samples.
        seed: Random seed for reproducibility.
    Returns: Tuple (X, y) where X is the feature matrix, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=(N, len(m)))
    ys = sample @ np.reshape(m, (-1, 1)) + b
    noise = rng.normal(loc=0.0, scale=50.0, size=ys.shape)
    return sample, (ys + noise).flatten()


# Test 3: Nonlinear exponential data
def nonlinear_data_generator1(m, b, range_, N, seed):
    """
    Generates nonlinear exponential data with noise.
    Args:
        m: Coefficient for the exponential term.
        b: Intercept of the function.
        range_: Tuple (low, high) for the range of X values.
        N: Number of samples.
        seed: Random seed for reproducibility.
    Returns: Tuple (X, y) where X is the feature array, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=N)
    ys = np.exp(m * sample) + b
    noise = rng.normal(loc=0.0, scale=0.5, size=N)
    return sample.reshape(-1, 1), ys + noise


# Test 4: Data with collinearity
def generate_collinear_data(range_, noise_scale, size, seed):
    """
    Generates data with collinear features and noise.
    Args:
        range_: Tuple (low, high) for the range of feature values.
        noise_scale: Standard deviation of noise added to the target.
        size: Tuple (n_samples, n_features) for the dataset size.
        seed: Random seed for reproducibility.
    Returns: Tuple (X, y) where X is the feature matrix, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=size)
    new_col = rng.normal(loc=0.0, scale=0.01, size=sample.shape[0]).reshape(-1, 1)
    new_sample = np.hstack((sample, new_col))
    coefficients = rng.integers(low=-10, high=10, size=(new_sample.shape[1], 1))
    ys = new_sample @ coefficients
    noise = rng.normal(loc=0.0, scale=noise_scale, size=ys.shape)
    return new_sample, (ys + noise).flatten()


# Test 5: Periodic data
def generate_periodic_data(period, amplitude, range_, noise_scale, size, seed):
    """
    Generates periodic data with sawtooth waveform.
    Args:
        period: Period of the waveform.
        amplitude: Amplitude of the waveform.
        range_: Tuple (low, high) for the range of X values.
        noise_scale: Standard deviation of noise added to the target.
        size: Number of samples.
        seed: Random seed for reproducibility.
    Returns: Tuple (X, y) where X is the feature array, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=size)
    ys = amplitude * sawtooth(2 * np.pi * sample / period - 1.47)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=ys.shape)
    return sample.reshape(-1, 1), ys + noise


# Test 6: Higher-dimensional nonlinear data
def generate_higher_dim_data(range_, noise_scale, size, seed):
    """
    Generates higher-dimensional nonlinear data with noise.
    Args:
        range_: Tuple (low, high) for the range of feature values.
        noise_scale: Standard deviation of noise added to the target.
        size: Tuple (n_samples, n_features) for the dataset size.
        seed: Random seed for reproducibility.
    Returns:Tuple (X, y) where X is the feature matrix, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=range_[0], high=range_[1], size=size)
    ys = sample[:, 0] ** 2 + sample[:, 1] ** 3 - np.linalg.norm(sample, axis=1)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=ys.shape)
    return sample, ys + noise


# Test 7: High collinearity with many features
def generate_high_collinear_data(n_features, size, seed):
    """
    Generates data with high collinearity and many features.
    Args:
        n_features: Number of highly collinear features.
        size: Number of samples.
        seed: Random seed for reproducibility.
    Returns:Tuple (X, y) where X is the feature matrix, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    base = rng.uniform(low=-5, high=5, size=size)
    collinear_data = np.array([base + rng.normal(0, 0.01, size) for _ in range(n_features)]).T
    coefficients = rng.uniform(low=1, high=10, size=n_features)
    ys = collinear_data @ coefficients + rng.normal(0, 0.1, size)
    return collinear_data, ys


# Test 8: Sparse, noisy, and collinear data
def generate_horrible_data(n_features, size, seed):
    """
    Generates extremely challenging data with sparsity, noise, and collinearity.
    Args:
        n_features: Number of features.
        size: Number of samples.
        seed: Random seed for reproducibility.
    Returns: Tuple (X, y) where X is the feature matrix, y is the target array.
    """
    rng = np.random.default_rng(seed=seed)
    base = rng.uniform(low=-5, high=5, size=size)
    collinear_data = np.array([base + rng.normal(0, 0.001, size) for _ in range(n_features)]).T
    sparsity_mask = rng.choice([0, 1], size=collinear_data.shape, p=[0.9, 0.1])  # 90% zeros
    collinear_data = collinear_data * sparsity_mask
    coefficients = rng.uniform(low=1, high=5, size=n_features)
    ys = collinear_data @ coefficients + rng.normal(0, 100, size)  # High noise
    return collinear_data, ys

