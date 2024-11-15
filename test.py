from modelSelection import *
from lib import *


def main():
    # Init settings
    args = params["test"]["general"]
    model = get_model(args["model"])
    metric = get_metric(args["metric"])
    train_X, train_y, test_X, test_y = get_data(args["data"])
    X = np.concatenate([train_X, test_X], axis=0)
    y = np.concatenate([train_y, test_y], axis=0)

    # k-Fold Cross-Validation
    args = params["test"]["k_fold_cross_validation"]
    k_fold_scores, k_fold_avg = k_fold_cross_validation(model, metric, X=X, y=y, k=args["k"], shuffle=args["shuffle"])
    print("k-Fold Cross-Validation Scores:", k_fold_scores)
    print("k-Fold Average Score:", k_fold_avg)

    # Compare with simple AIC
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    y_pred = model.predict(test_X)
    print("AIC Scores: ", AIC(X=test_X, y=test_y, y_pred=y_pred))

    # bootstrapping
    args = params["test"]["bootstrapping"]
    bootstrap_scores, bootstrap_avg = bootstrapping(model, metric, X=X, y=y, s=args["size"], epochs=args["epochs"])
    print("Bootstrap Scores:", bootstrap_scores)
    print("Bootstrap Average Score:", bootstrap_avg)


if __name__ == "__main__":
    main()
