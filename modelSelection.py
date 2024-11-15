from lib import *


def k_fold_cross_validation(model, metric, X: np.ndarray, y: np.ndarray, k: int, shuffle: bool):
    """
    k_fold_cross_validation()
    This function is used to validate the model using k-fold cross validation method

    :param model: The statistic model to test
    :param metric: The metric to measure the model performance
    :param X: The training feature vectors
    :param y: The training labels
    :param k: Number of folds
    :param shuffle: Whether to shuffle the data before the operation
    :return: metric scores of folds and their average score
    """
    scores = []
    n = X.shape[0]  # Size of sample
    fold_size = n // k  # Size of each fold

    if shuffle:  # shuffle the data
        indices = np.arange(n)
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

    for i in range(k):
        # Get indices(start, end) of the current validation fold
        start, end = i * fold_size, (i + 1) * fold_size

        # Get validation set
        X_val, y_val = X[start:end], y[start:end]

        # Get train set
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        # Train the model
        model.fit(X_train, y_train)

        # Get predicted data
        y_predicted = model.predict(X_val)

        # Compute accuracy rate
        score = metric(y_val, y_predicted)

        # Add to score list
        scores.append(score)

    return scores, np.average(scores)


def bootstrapping(model, metric, X: np.ndarray, y: np.ndarray, s: int, epochs: int):
    """
    bootstrapping()
    This function performs bootstrapping

    :param model: The statistic model to test
    :param metric: The metric to measure the model performance
    :param X: The training feature vectors
    :param y: The training labels
    :param s: The size of a train dataset
    :param epochs: The number of trials to perform train/verify its performance
    :return: metric scores of trials and their average score
    """
    scores = []
    n = X.shape[0]  # Size of dataset

    for _ in range(epochs):
        # Pick train set data by sampling of size s
        indices = np.random.choice(range(n), size=s, replace=True)
        X_train, y_train = X[indices], y[indices]

        # The others will be assigned as validation set
        out_of_sample = [i for i in range(n) if i not in indices]
        X_val, y_val = X[out_of_sample], y[out_of_sample]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if len(out_of_sample) > 0:  # Avoid cases with no out-of-sample data
            score = metric(y_val, y_pred)
            scores.append(score)

    average_score = np.mean(scores)
    return scores, average_score
