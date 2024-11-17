from modelSelection import *


def test_k_Fold_CV(model, metric, X: np.ndarray, y: np.ndarray, ks: list[int], shuffle: bool):
    for k in ks:
        k_fold_scores, k_fold_avg = k_fold_cross_validation(model, metric, X=X, y=y, k=k, shuffle=shuffle)
        print(f"k-Fold Cross-Validation Scores:\n\t{k_fold_scores}")
        print(f"k-Fold Average Score: {k_fold_avg}")


def test_bootstrapping(model, metric, X: np.ndarray, y: np.ndarray, ss: list[int], epochs_list: list[int]):
    for s in ss:
        for epochs in epochs_list:
            bootstrap_scores, bootstrap_avg = bootstrapping(model, metric, X=X, y=y, s=s, epochs=epochs)
            print(f"Bootstrap Scores (Pick first 5 out of {len(bootstrap_scores)}):\n\t{bootstrap_scores[:5]}")
            print(f"Bootstrap Score range: [{min(bootstrap_scores)}, {max(bootstrap_scores)}]")
            print(f"Bootstrap Median Score: {np.median(bootstrap_scores)}")
            print(f"Bootstrap Average Score: {bootstrap_avg}")


def test_AIC(model, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    print(f"\t(Comparison) AIC Score: {AIC(X=test_X, y=test_y, y_pred=y_pred)}")


def main():
    param = get_param('./params/param_multi.json')
    # Init params
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

        # Get dataset
        X, y, train_X, train_y, test_X, test_y = get_data(args_g["data"], args)

        print("-" * 121)

        if args_g["activate"]["k_fold_CV"]:
            # k-Fold Cross-Validation
            print("[Test] K-Fold Cross-Validation")
            args_k = param["test"]["k_fold_cross_validation"]

            # Test K-Fold CV
            test_k_Fold_CV(model, metric, X, y, ks=args_k["k"], shuffle=args_k["shuffle"])

            # Compare with simple AIC
            test_AIC(model, train_X, train_y, test_X, test_y)

            print("-" * 121)

        if args_g["activate"]["bootstrapping"]:
            # bootstrapping
            print("[Test] Bootstrapping")
            args_b = param["test"]["bootstrapping"]
            test_bootstrapping(model, metric, X, y, ss=args_b["size"], epochs_list=args_b["epochs"])
        print(f"{'=' * 52} [Test {i:2d}] End   {'=' * 52}")

        # Increase index number
        i += 1
        print("")


if __name__ == "__main__":
    main()
