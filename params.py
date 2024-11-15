# All User parameter can be adjusted from here.

"""
 Test configuration
"""

params = {
    "test": {
        "general": {
            "model": "LinearRegression",  # [options] "LinearRegression(default)", "LogisticRegression"
            "metric": "MSE",  # [options] "MSE(default)", "Accuracy score"
            "data": "synthetic",  # [options] 'kaggle'(default), 'file', 'synthetic', 'multi-collinear'
        },
        "k_fold_cross_validation": {
            "k": 5,  # Number of folds
            "shuffle": True  # shuffle: Whether to shuffle the data before the operation
        },
        "bootstrapping": {
            "size": 50,  # The size of a train dataset
            "epochs": 100  # The number of trials to perform train/verify its performance
        }
    },
    "data": {
        # User defined params for kaggle data import setting
        "kaggle": {
            "test_ratio": 0.3  # Portion of the test set from total sample.
        },
        # User defined params for data import setting
        "file": {
            "file_name": "small_test.csv",  # File path to read
            "test_ratio": 0.3  # Portion of the test set from total sample.
        },
        # User defined params for synthetic data creation setting
        "synthetic": {
            "size": 1000,  # Number of samples to create
            "dimension": 9,  # Number of features of the data
            "noise_std": 0.00,  # Noise scale to check durability of the model
            "random_state": 42,  # Random seed
            "test_ratio": 0.3  # Portion of the test set from total sample.
        },
        # User defined params for multi-collinear data creation setting
        "multi-collinear": {
            "size": 100,  # Number of samples to create
            "dimension": 3,  # Number of features of the data
            "correlation": 1,  # Correlation coefficient between features
            "noise_std": 0.01,  # Noise scale to check durability of the model
            "random_state": 42,  # Random seed
            "test_ratio": 0.3  # Portion of the test set from total sample.
        }
    }
}
