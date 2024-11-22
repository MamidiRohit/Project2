import numpy as np

def bootstrap(model, X, y, B=100, metric="mse"):
    n = len(X)
    scores = []

    for _ in range(B):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        X_sample, y_sample = X[indices], y[indices]
        
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        
        if metric == "mse":
            score = np.mean((y_sample - y_pred) ** 2)
        elif metric == "mae":
            score = np.mean(np.abs(y_sample - y_pred))
        else:
            raise ValueError("Unsupported metric.")
        
        scores.append(score)

    return scores, np.mean(scores)
