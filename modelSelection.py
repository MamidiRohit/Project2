from lib import *


def k_fold_cross_validation(model, metric, X: np.ndarray, y: np.ndarray, k: int, shuffle: bool):
    """
    k_fold_cross_validation()
    This function validates the model using the k-fold cross-validation method.

    :param model: The statistical model to test.
    :param metric: The metric function to measure the model's performance.
    :param X: The feature matrix for training.
    :param y: The target labels for training.
    :param k: The number of folds to divide the data into.
    :param shuffle: Whether to shuffle the data before splitting into folds.
    :return: A tuple containing the list of metric scores for each fold and their average score.
    """
    scores = []
    n = X.shape[0]  # Total number of samples
    fold_size = n // k   # Number of samples per fold

    if shuffle:  # Shuffle the data
        indices = np.arange(n)
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

    for i in range(k):
        # Define the start and end indices of the current validation fold
        start, end = i * fold_size, (i + 1) * fold_size

        # Extract the validation set
        X_val, y_val = X[start:end], y[start:end]

        # Extract the training set by excluding the current fold
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_predicted = model.predict(X_val)

        # Calculate the score using the metric function
        score = metric(y_val, y_predicted)

        # Append the score to the list
        scores.append(score)

    # Return the list of scores and their average
    return scores, np.average(scores)


def bootstrapping(model, metric, X: np.ndarray, y: np.ndarray, s: int, epochs: int):
    """
    bootstrapping()
    This function performs bootstrapping to evaluate the model's performance.

    :param model: The statistical model to test.
    :param metric: The metric function to measure the model's performance.
    :param X: The feature matrix for training.
    :param y: The target labels for training.
    :param s: The size of the training dataset for each bootstrap sample.
    :param epochs: The number of bootstrap iterations to perform.
    :return: A tuple containing the list of metric scores for each iteration and their average score.
    """
    scores = []
    n = X.shape[0]  # Total number of samples

    for _ in range(epochs):
        # Randomly sample 's(=size of sample)' indices with replacement to create the training set
        indices = np.random.choice(range(n), size=s, replace=True)
        X_train, y_train = X[indices], y[indices]

        # Use the remaining data as the validation set
        out_of_sample = [i for i in range(n) if i not in indices]
        X_val, y_val = X[out_of_sample], y[out_of_sample]

        # Train the model on the bootstrap sample
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = model.predict(X_val)

        # Calculate the score if there is any validation data
        if len(out_of_sample) > 0:
            score = metric(y_val, y_pred)
            scores.append(score)

    # Calculate the average score across all iterations
    average_score = np.mean(scores)
    return scores, average_score
