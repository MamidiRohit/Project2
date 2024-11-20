# import sys
from modelSelection import *


def test_k_Fold_CV(model, metric, X: np.ndarray, y: np.ndarray, ks: list[int], shuffle: bool):
    """
    test_k_Fold_CV()
    This function tests the k-fold cross-validation implementation with different values of k.

    :param model: The statistical model to be validated.
    :param metric: The metric function used to evaluate model performance.
    :param X: The feature matrix.
    :param y: The target labels.
    :param ks: A list of k values to test (number of folds).
    :param shuffle: Whether to shuffle the data before splitting into folds.
    """
    for k in ks:
        k_fold_scores, k_fold_avg = k_fold_cross_validation(model, metric, X=X, y=y, k=k, shuffle=shuffle)
        print(f"k-Fold Cross-Validation Scores:\n\t{k_fold_scores}")
        print(f"k-Fold Average Score: {k_fold_avg}")


def test_bootstrapping(model, metric, X: np.ndarray, y: np.ndarray, ss: list[int], epochs_list: list[int]):
    """
    test_bootstrapping()
    This function tests the bootstrapping implementation with different sample sizes and epochs.

    :param model: The statistical model to be tested.
    :param metric: The metric function used to evaluate model performance.
    :param X: The feature matrix.
    :param y: The target labels.
    :param ss: A list of sample sizes for the bootstrapping training set.
    :param epochs_list: A list of epoch values to determine the number of iterations for bootstrapping.
    """
    for s in ss:
        for epochs in epochs_list:
            bootstrap_scores, bootstrap_avg = bootstrapping(model, metric, X=X, y=y, s=s, epochs=epochs)
            print(f"Bootstrap Scores (Pick first 5 out of {len(bootstrap_scores)}):\n\t{bootstrap_scores[:5]}")
            print(f"Bootstrap Score range: [{min(bootstrap_scores)}, {max(bootstrap_scores)}]")
            print(f"Bootstrap Median Score: {np.median(bootstrap_scores)}")
            print(f"Bootstrap Average Score: {bootstrap_avg}")


def test_AIC(model, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
    """
    test_AIC()
    This function tests the AIC (Akaike Information Criterion) computation for the given model.

    :param model: The trained statistical model.
    :param train_X: Training feature matrix.
    :param train_y: Training labels.
    :param test_X: Test feature matrix.
    :param test_y: Test labels.
    """
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    print(f"\t(Comparison) AIC Score: {AIC(X=test_X, y=test_y, y_pred=y_pred)}")


def main(file_path: str):
    """
    main()
    This function is an entry point for the test suite. Loads parameters, initializes models,
    and tests k-fold cross-validation with AIC performance and bootstrapping.

    :param file_path: Path to the JSON configuration file containing test parameters.
    """
    param = get_param(file_path)

    # Initialize global parameters
    print(f"{'*' * 52} Global Setting {'*' * 52}")
    args_g = param["test"]["general"]
    print(f"Description:\n\t{param['description']}")
    model = get_model(args_g["model"])
    metric = get_metric(args_g["metric"])
    print(f"Data Type: {args_g['data']}")
    print("*" * 121, "\n")

    i = 0
    while i < len(param["data"][args_g["data"]]):
        print(f"{'=' * 52} [Test {i:2d}] Start {'=' * 52}")
        args = param["data"][args_g["data"]][i]
        print(f"Data Parameters:\n\t{args}")

        # Load dataset
        X, y, train_X, train_y, test_X, test_y = get_data(args_g["data"], args)

        print("-" * 121)

        if args_g["activate"]["k_fold_CV"]:
            # k-Fold Cross-Validation
            print("[Test] K-Fold Cross-Validation")
            args_k = param["test"]["k_fold_cross_validation"]

            # Perform K-Fold CV testing
            test_k_Fold_CV(model, metric, X, y, ks=args_k["k"], shuffle=args_k["shuffle"])

            # Compare results with AIC
            test_AIC(model, train_X, train_y, test_X, test_y)

            print("-" * 121)

        if args_g["activate"]["bootstrapping"]:
            # Bootstrapping Testing
            print("[Test] Bootstrapping")
            args_b = param["test"]["bootstrapping"]
            test_bootstrapping(model, metric, X, y, ss=args_b["size"], epochs_list=args_b["epochs"])
        print(f"{'=' * 52} [Test {i:2d}] End   {'=' * 52}")

        # Increase index number
        i += 1
        print("")


if __name__ == "__main__":
    # args = sys.argv
    args = ["QuickStart", "./params/param_single.json"]
    if len(args) > 1:
        main(args[1])
    else:
        print("[Warning] No parameter configuration (file path with param_*.json) provided!")
        print("[Info] Program terminated.")
