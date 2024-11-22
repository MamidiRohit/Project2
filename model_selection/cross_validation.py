import numpy as np

def k_fold_cv(model, X, y, k=5, metric="mse"):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    scores = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        val_idx = folds[i]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if metric == "mse":
            score = np.mean((y_val - y_pred) ** 2)
        elif metric == "mae":
            score = np.mean(np.abs(y_val - y_pred))
        else:
            raise ValueError("Unsupported metric.")
        scores.append(score)

    return np.mean(scores)
