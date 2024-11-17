# All User parameter can be adjusted from here.

"""
 Test configuration
"""

params = {
    "test": {
        "general": {
            "activate": {"k_fold_CV": True, "bootstrapping": True},  # True to execute the target test
            "model": "LinearRegression",  # [options] "LinearRegression(default)", "LogisticRegression"
            "metric": "MSE",  # [options] "MSE(default)", "Accuracy score"
            "data": "generate",  # [options] 'kaggle'(default), 'file', 'generate'
        },
        "k_fold_cross_validation": {
            "k": [5],  # Number of folds
            "shuffle": True  # shuffle: Whether to shuffle the data before the operation
        },
        "bootstrapping": {
            "size": [50],  # The size of a train dataset
            "epochs": [100]  # The number of trials to perform train/verify its performance
        }
    },
    "data": {
        # User defined params for data import setting
        "file": [
            {
                "file_name": "small_test.csv",  # File path to read
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
        ],
        # User defined params data creation setting
        "generate": [
            {  # Default dataset with strict linear relationship
                "size": 100,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0,  # Correlation coefficient between features
                "noise_std": 0.0,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Larger dataset with strict linear relationship
                "size": 500,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0,  # Correlation coefficient between features
                "noise_std": 0.00,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Default dataset with some noise
                "size": 100,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0,  # Correlation coefficient between features
                "noise_std": 0.01,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Larger dataset with some noise
                "size": 500,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0,  # Correlation coefficient between features
                "noise_std": 0.01,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Default dataset with high correlation
                "size": 100,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0.9,  # Correlation coefficient between features
                "noise_std": 0.0,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Larger dataset with high correlation
                "size": 100,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0.9,  # Correlation coefficient between features
                "noise_std": 0.0,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Default dataset with some noise and high correlation
                "size": 100,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0.9,  # Correlation coefficient between features
                "noise_std": 0.01,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            },
            {  # Larger dataset with some noise and high correlation
                "size": 100,  # Number of samples to create
                "dimension": 10,  # Number of features of the data
                "correlation": 0.9,  # Correlation coefficient between features
                "noise_std": 0.01,  # Noise scale to check durability of the model
                "random_state": 42,  # Random seed
                "test_ratio": 0.25  # Portion of the test set from total sample.
            }
        ]
    }
}
