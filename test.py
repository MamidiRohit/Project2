import sys
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
    :return: A list of results, where each result is a list containing:
             [k (fold count), shuffle (bool), scores (list of fold scores), average (mean score)].
    """
    results = []
    for k in ks:
        print(f"\tk-Fold Cross-Validation K-value: {k}")
        print(f"\tk-Fold Cross-Validation shuffling: {shuffle}")
        scores, average = k_fold_cross_validation(model, metric, X=X, y=y, k=k, shuffle=shuffle)
        print(f"\tk-Fold Cross-Validation Scores:\n\t\t{scores}")
        print(f"\tk-Fold Average Score: {round(average, 5)}")
        results.append([k, shuffle, round(average, 5)])
        print("")
    return results


def test_bootstrapping(model, metric, X: np.ndarray, y: np.ndarray, ss: list[int], epochs_list: list[int]) -> list:
    """
    test_bootstrapping()
    This function tests the bootstrapping implementation with different sample sizes and epochs.

    :param model: The statistical model to be tested.
    :param metric: The metric function used to evaluate model performance.
    :param X: The feature matrix.
    :param y: The target labels.
    :param ss: A list of sample sizes for the bootstrapping training set.
    :param epochs_list: A list of epoch values to determine the number of iterations for bootstrapping.
    :return: A list of results, where each result is a list containing:
             [s (sample size), epochs, scores (list of metric scores for each epoch), average (mean score)].

    """
    results = []
    for s in ss:
        for epochs in epochs_list:
            print(f"\tBootstrap sample size: {s}")
            print(f"\tBootstrap epochs: {epochs}")
            scores, average = bootstrapping(model, metric, X=X, y=y, s=s, epochs=epochs)
            print(f"\tBootstrap Scores (Pick first 5 out of {len(scores)}):\n\t\t{scores[:5]}")
            print(f"\tBootstrap Score range: [{round(min(scores), 5)}, {round(max(scores), 5)}]")
            print(f"\tBootstrap Median Score: {round(np.median(scores), 5)}")
            print(f"\tBootstrap Average Score: {round(average, 5)}")
            results.append([s, epochs, round(average, 5)])
            print("")

    return results


def test_AIC(model, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
    """
    test_AIC()
    This function tests the AIC (Akaike Information Criterion) computation for the given model.

    :param model: The trained statistical model.
    :param train_X: Training feature matrix.
    :param train_y: Training labels.
    :param test_X: Test feature matrix.
    :param test_y: Test labels.
    :return: AIC value.
    """
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    aic = AIC(X=test_X, y=test_y, y_pred=y_pred)
    return round(aic, 5)


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

    # Store global results
    results_K_fold = []
    result_boostrap = []

    i = 0
    while i < len(param["data"][args_g["data"]]):
        print(f"{'=' * 52} [Test {i:2d}] Start {'=' * 52}")
        args_d = param["data"][args_g["data"]][i]
        print(f"Data Parameters:\n\t{args_d}")

        # Load dataset
        X, y, train_X, train_y, test_X, test_y = get_data(args_g["data"], args_d)

        print("-" * 121)

        # Compute AIC in advance
        aic = float(test_AIC(model, train_X, train_y, test_X, test_y))

        if args_g["activate"]["k_fold_CV"]:
            # k-Fold Cross-Validation
            print("[Test] K-Fold Cross-Validation")
            args_k = param["test"]["k_fold_cross_validation"]

            # Perform K-Fold CV testing
            results = test_k_Fold_CV(model, metric, X, y, ks=args_k["k"], shuffle=args_k["shuffle"])

            # Append test results to global list results_K_fold
            if args_g["data"] == 'generate':
                for result in results:
                    result.insert(0, args_d["noise_std"])
                    result.insert(0, args_d["correlation"])
                    result.insert(0, args_d["dimension"])
                    result.insert(0, args_d["size"])
                    result.append(aic)
                    results_K_fold.append(result)

            print("-" * 121)

        if args_g["activate"]["bootstrapping"]:
            # Bootstrapping Testing
            print("[Test] Bootstrapping")
            args_b = param["test"]["bootstrapping"]
            print(f"Bootstrapping Parameters:\n\t{args_b}")

            # Perform Bootstrapping testing
            results = test_bootstrapping(model, metric, X, y, ss=args_b["size"], epochs_list=args_b["epochs"])
            # Append test results to global list result_boostrap
            if args_g["data"] == 'generate':
                for result in results:
                    result.insert(0, args_d["noise_std"])
                    result.insert(0, args_d["correlation"])
                    result.insert(0, args_d["dimension"])
                    result.insert(0, args_d["size"])
                    result.append(aic)
                    result_boostrap.append(result)

        print("-" * 121)

        # Show the comparative score from AIC
        print(f"[Test] AIC Score: {round(aic, 5)}")

        print(f"{'=' * 52} [Test {i:2d}] End   {'=' * 52}")

        # Increase index number
        i += 1
        print("")

    # Visualization and file write only support for the dataset type 'generate'!
    if args_g["data"] == 'generate':
        args_a = param["test"]['analysis']
        # Visualize if activated & only if results are many then just 1
        if args_a["visualize"]["k_fold_CV"]["activate"] and len(results_K_fold) > 1:
            visualize(results_K_fold, "k-fold Cross Validation", args_a["visualize"]["k_fold_CV"]["label_X"])

        if args_a["visualize"]["bootstrapping"]["activate"] and len(result_boostrap) > 1:
            visualize(result_boostrap, "Bootstrapping", args_a["visualize"]["bootstrapping"]["label_X"])

        # Write data if activated & only if results are many then just 1
        if args_a["write"]["k_fold_CV"]["activate"] and len(results_K_fold) > 1:
            file_path = args_a["write"]["k_fold_CV"]["file_path"]
            header = args_a["write"]["k_fold_CV"]["header"]
            write(file_path, results_K_fold, header)

        if args_a["write"]["bootstrapping"]["activate"] and len(result_boostrap) > 1:
            file_path = args_a["write"]["bootstrapping"]["file_path"]
            header = args_a["write"]["bootstrapping"]["header"]
            write(file_path, result_boostrap, header)


if __name__ == "__main__":
    arguments = sys.argv
    # Test code below
    # arguments = ["QuickStart", "./params/param_single.json"]
    # arguments = ["QuickStart", "./params/param_multi.json"]
    # arguments = ["QuickStart", "./params/param_k_fold.json"]
    # arguments = ["QuickStart", "./params/param_bootstrap.json"]
    # arguments = ["QuickStart", "./params/test_size.json"]
    # arguments = ["QuickStart", "./params/test_correlation.json"]

    if len(arguments) > 1:
        main(arguments[1])
    else:
        print("[Warning] No parameter configuration (file path with param_*.json) provided!")
        print("[Info] Program terminated.")
