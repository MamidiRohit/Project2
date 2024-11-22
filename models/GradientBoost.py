import numpy as np

class GradientBoostingTree:
    
    """
    A custom implementation of Gradient Boosting for regression tasks using decision trees.

    Parameters:
    - n_estimators: int, default=100
        The number of boosting stages to perform (number of trees).
    - learning_rate: float, default=0.1
        Shrinks the contribution of each tree by this value. Larger values may lead to overfitting.
    - max_depth: int, default=3
        The maximum depth of individual regression trees.
    - loss: str, default="squared_error"
        The loss function to optimize. Currently supports "squared_error" for regression.

    Attributes:
    - trees: list
        A list of fitted regression trees forming the boosting ensemble.
    - initial_prediction: float
        The initial model prediction (mean of the target values).

    Methods:
    - initialize_model_parameters(y):
        Initializes the model parameters, typically the mean of the target values for squared error loss.
    
    - loss_gradient(y, pred):
        Computes the gradient of the loss function with respect to predictions.
    
    - fit_ensemble_tree(X, residuals):
        Fits a single regression tree to the given residuals.
    
    - construct_tree(X, y, depth):
        Recursively constructs a regression tree by finding the best splits.
    
    - predict_tree(x, tree):
        Predicts the output for a single sample using the fitted tree structure.
    
    - fit(X, y):
        Fits the Gradient Boosting model on the given training data.
    
    - predict(X):
        Predicts the target values for the input features using the ensemble of trees.
    
    - r2_score_manual(y_true, y_pred):
        Computes the R² (coefficient of determination) score manually.
    
    - mae_manual(y_true, y_pred):
        Computes the Mean Absolute Error (MAE) manually.
    
    - rmse_manual(y_true, y_pred):
        Computes the Root Mean Squared Error (RMSE) manually.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss="squared_error"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.trees = []
        self.initial_prediction = None

    def initialize_model_parameters(self, y):
        return np.mean(y)

    def loss_gradient(self, y, pred):
        return y - pred  

    def fit_ensemble_tree(self, X, residuals):
        tree = self.construct_tree(X, residuals, depth=0)
        return tree

    def construct_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"value": np.mean(y)}

        n_samples, n_features = X.shape
        best_split = None
        min_error = float("inf")

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                left_mean = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_mean = np.mean(y[right_mask]) if np.any(right_mask) else 0

                error = np.sum((y[left_mask] - left_mean) ** 2) + np.sum((y[right_mask] - right_mean) ** 2)

                if error < min_error:
                    min_error = error
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }

        left_tree = self.construct_tree(X[best_split["left_mask"]], y[best_split["left_mask"]], depth + 1)
        right_tree = self.construct_tree(X[best_split["right_mask"]], y[best_split["right_mask"]], depth + 1)

        return {
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def predict_tree(self, x, tree):
        if "value" in tree:
            return tree["value"]

        if x[tree["feature_index"]] <= tree["threshold"]:
            return self.predict_tree(x, tree["left"])
        else:
            return self.predict_tree(x, tree["right"])

    def fit(self, X, y):
        self.initial_prediction = self.initialize_model_parameters(y)
        pred = np.full(y.shape, self.initial_prediction, dtype=np.float64)

        for _ in range(self.n_estimators):
            residuals = self.loss_gradient(y, pred)
            tree = self.fit_ensemble_tree(X, residuals)
            self.trees.append(tree)
            pred += self.learning_rate * np.array([self.predict_tree(x, tree) for x in X])

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)

        for tree in self.trees:
            pred += self.learning_rate * np.array([self.predict_tree(x, tree) for x in X])

        return pred

    
    def r2_score_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        y_mean = np.mean(y_true)
        tss = np.sum((y_true - y_mean) ** 2)
        rss = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (rss / tss)
        return r2

    def mae_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def rmse_manual(self, y_true, y_pred):
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse