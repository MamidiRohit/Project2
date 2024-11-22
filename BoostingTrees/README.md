# Project 2

Select one of the following two options:

## Boosting Trees

Implement the gradient-boosting tree algorithm (with the usual fit-predict interface) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1.

Put your README below. Answer the following questions.

Project 2
Group Name: A20560777
Group Members:
Pooja Pranavi Nalamothu (CWID: A20560777)
Gradient Boosting Tree
1.	What does the model you have implemented do, and when should it be used?
Gradient Boosting Tree (GBT) is an algorithm used in supporting machine learning for regression problems. It develops an additive model that is constructed sequentially through using predictions from decisions trees which each step aims at capturing residuals of the trees employed earlier on. This approach makes it flexible for use when dealing with large amounts of datasets, and good for recognizing non-linear patterns.
When to use it:
•	When you are working with numeric features and your target variable is a continuous variable (regression problem).
•	When it is difficult to express a dependence between an independent variable and a target by reference to the linear equation of regression.
•	When it is necessary to understand, which features affect models’ decisions in decision trees.
•	It is most appropriate for datasets of considerable size but not for datasets to which a large number of observations is attributed as the GBT models can be resource demanding when used on large datasets.
2.	How did you test your model to determine if it is working reasonably correctly?
Synthetic Dataset:
To further evaluate this model, we tested it on a synthetic data which was created using a linear weighted sum of all the features with some noise added. The use of the model was an efficient approach to reducing the Mean Squared Error (MSE) and proposed values that were near the actual numbers.
Real Datasets:
Applied the model to various real-world datasets:
•	Energy Efficiency Dataset: Predicted heating loads in buildings.
•	Medical Cost Dataset: Forecast of insurance to charges depending on demographic and health characteristics.
•	Auto MPG Dataset: In these models fuel efficiency in terms of car’s MPG has been predicted based on vehicle characteristics.
•	Wine Quality Dataset: To reflect the quality of wine three general metrics of quality where used to predict the quality scores of the wine given in function of the chemical features.
Validation Metrics:

•	Summarized the model’s validation by manually calculating the Mean Squared Error (MSE) for accuracy’s sake, not without prebuilt libraries.
•	Calculated probabilities of the light level to ensure that they were realistic, that is, within the optimal range of the target values and consistent with the data trends.
Debugging:
•	Measures the numbers of input features, targets, and prediction to standardization among datasets.
•	The findings were then analyzed relative to baseline models such as mean predictions to establish improved outcomes.

3.	What parameters have you exposed to users of your implementation in order to tune performance?
The following parameters are exposed in the implementation:
n_estimators: Number of decision trees to train.
•	Higher values allow better learning but increase training time.
•	Default: 50.
learning_rate: Determines the contribution of each tree to the final prediction.
•	Smaller values prevent overfitting but require more trees.
•	Default: 0.1.
max_depth: Maximum depth of individual trees.
•	Controls the complexity of each tree and prevents overfitting.
•	Default: 3.
loss: Loss function for computing gradients.
•	Current implementation supports squared error for regression

4.	Are there specific inputs that your implementation has trouble with?
Categorical Features:
It is also important to note that the current development does not directly cater for categorical variables. Such features must be encoded by users using one-hot encoding or label encoding.
Large Datasets:
An important limitation of handling very large datasets (e.g., millions of samples) involves high computational costs, owing to the sequential structure of gradient boosting.
Imbalanced Data:
In the case of having highly skewed distributions in a target variable, it becomes hard for a model to predict an event that rarely occurs.
High-Dimensional Features:
In particular, if the number of features is large enough and varies, for example, in thousands, then the splits of the decision tree are also large enough and may not generalize well, so the model will begin to over-fit.
Given more time, could you work around these, or is it fundamental?
Yes, most of these issues can be addressed with additional time:
•	Categorical Handling: One of the modifications carried out is to try to incorporate directly splits of categorical variables in the decision tree.
•	Scalability: The use of parallelization, or incremental boosting technique such as chunking of the data will go a long way in ensuring that the techniques do not give out negative results.
•	Loss Function Extension: It will also enhance its support for other types of losses such as absolute error, Huber, or type of classification loss.
However, some challenges such as overfitting when working with high dimensional features are all inherent characteristics of a decision tree model and may only be solved either with external data preprocessing tools or by use of other architectures of the model.

How to run the code
Step 1:
Clone or download the code: Ensure all the datasets are in the correct paths as mentioned in the code.
Step 2:
Install necessary dependencies
 
Step 3:
Run the script
 
Step 4:
Ensure all the datasets are properly loaded and called
Note: Update paths in the script, if required.
Basic Usage:
•	For Synthetic Data, you can run this code directly
CODE:
import numpy as np

class DecisionTreeRegressor:
    """
    A simple decision tree regressor for fitting residuals.
    """
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def _split(self, X, y):
        """
        Find the best split for a dataset.
        """
        best_split = {"feature": None, "threshold": None, "loss": float("inf")}
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_residuals = y[left_mask]
                right_residuals = y[right_mask]
                
                # Mean squared error as loss
                loss = (
                    np.sum((left_residuals - np.mean(left_residuals)) ** 2) +
                    np.sum((right_residuals - np.mean(right_residuals)) ** 2)
                )
                
                if loss < best_split["loss"]:
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "loss": loss,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }
        
        return best_split

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        if depth >= self.max_depth or len(set(y)) == 1:
            return {"value": np.mean(y)}

        split = self._split(X, y)
        if split["feature"] is None:
            return {"value": np.mean(y)}

        left_tree = self._build_tree(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right_tree = self._build_tree(X[split["right_mask"]], y[split["right_mask"]], depth + 1)

        return {
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_one(self, x, tree):
        """
        Predict a single sample using the tree.
        """
        if "value" in tree:
            return tree["value"]
        
        feature = tree["feature"]
        threshold = tree["threshold"]

        if x[feature] <= threshold:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class GradientBoostingTree:
    """
    Gradient Boosting Tree implementation with explicit gamma calculation.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss="squared_error"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_prediction = None
        self.loss = loss

    def _gradient(self, y, y_pred):
        """
        Compute the gradient of the loss function.
        """
        if self.loss == "squared_error":
            return y - y_pred
        raise ValueError("Unsupported loss function")

    def _gamma(self, residuals, region):
        """
        Compute the optimal gamma for a region as per Equation (10.30).
        """
        return np.mean(residuals[region])

    def fit(self, X, y):
        """
        Train the gradient boosting tree model.
        """
        self.init_prediction = np.mean(y)  # Start with the mean prediction
        predictions = np.full_like(y, self.init_prediction, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradients)
            residuals = self._gradient(y, predictions)

            # Train a decision tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions with the tree's contribution
            tree_predictions = tree.predict(X)

            for region in np.unique(tree_predictions):
                mask = tree_predictions == region
                gamma = self._gamma(residuals, mask)
                predictions[mask] += self.learning_rate * gamma

    def predict(self, X):
        """
        Predict target values for input data X.
        """
        predictions = np.full((X.shape[0],), self.init_prediction, dtype=np.float64)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions


# Example Usage
if __name__ == "__main__":
    # Import necessary libraries
    import numpy as np

    # Generate synthetic regression data
    def make_synthetic_regression(n_samples=100, n_features=7, noise=0.1, random_state=42):
        np.random.seed(random_state)
        X = np.random.rand(n_samples, n_features)  # Features: random values in [0, 1]
        coefficients = np.random.rand(n_features)  # Random coefficients for linear relation
        y = X @ coefficients + noise * np.random.randn(n_samples)  # Linear relationship + noise
        return X, y

    # Compute mean squared error manually
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Generate data
    X, y = make_synthetic_regression(n_samples=100, n_features=7, noise=0.1, random_state=42)
    y = y / np.std(y)  # Normalize target for simplicity

    # Train Gradient Boosting Tree
    model = GradientBoostingTree(n_estimators=50, learning_rate=0.1, max_depth=3, loss="squared_error")
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    # Evaluate
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse:.4f}")

    print("Predictions for new data:", predictions[:10])  # Display first 10 predictions

•	For other datasets
Ensure the correct dataset is loaded and is in the correct directory.
Define X and y for the corresponding features and target variables.
Once you've modified the file path, you can use pytest to run all tests, including real-world datasets and synthetic dataset
•	Open a terminal in the directory containing your test script.
•	Run Pytest with the following command: pytest test_datasetname.py