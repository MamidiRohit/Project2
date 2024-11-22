import numpy
from models.Decision_Tree import DecisionTree

#Computing the sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

#Computing the softmax activation function along a specified axis.
def softmax(x, axis=1):
    shiftx = x - numpy.max(x, axis=axis, keepdims=True)
    exps = numpy.exp(shiftx)
    return exps / numpy.sum(exps, axis=axis, keepdims=True)

class GradientBoostedTree():
    # Initializing parameters for the gradient boosted tree model
    # B: Number of boosting rounds (trees to build)
    # lmd: Learning rate to scale contributions of each tree
    # max_depth: Maximum depth of each decision tree
    # min_sample_leaf: Minimum number of samples per leaf node
    # min_information_gain: Minimum information gain to split a node
    # loss: Loss function used for optimization ("square_error" or classification)
    def __init__(self, B=100, lmd=0.1, max_depth=3, min_sample_leaf=5, min_information_gain=0.0, loss="logistic"):
        self.B = B
        self.lmd = lmd
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.min_information_gain = min_information_gain
        self.loss = loss
        self.tree_list = []
        self.n_classes = None

    #Training the gradient boosted tree model on the provided dataset
    def fit(self, data):
        X, y = data[:, :-1], data[:, -1]
        self.n_classes = len(numpy.unique(y))

        #Handling regression (square error loss)
        if self.loss == "square_error":
            self.initial_predictions = numpy.full(X.shape[0], numpy.mean(y))
            current_predictions = self.initial_predictions.copy()

            for i in range(self.B):
                #Computing residuals
                residuals = y - current_predictions
                tree = DecisionTree(self.max_depth, self.min_sample_leaf, self.min_information_gain, "logistic")
                #Combining features with residuals as target
                tree_data = numpy.column_stack((X, residuals))
                #Training decision tree on residuals
                fitted_tree = tree.fit(tree_data)
                #Storing the tree
                self.tree_list.append(fitted_tree)
                #Updating predictions
                current_predictions += self.lmd * fitted_tree.predict(X)

        #Handling binary classification
        elif self.n_classes == 2:
            self.initial_predictions = numpy.zeros(X.shape[0]) # Starting with all-zero predictions
            current_predictions = self.initial_predictions.copy()

            for i in range(self.B):
                #Calculating probabilities using sigmoid function.
                p = sigmoid(current_predictions)
                residuals = y - p
                tree = DecisionTree(self.max_depth, self.min_sample_leaf, self.min_information_gain, "logistic")
                tree_data = numpy.column_stack((X, residuals))
                fitted_tree = tree.fit(tree_data)
                self.tree_list.append(fitted_tree)
                current_predictions += self.lmd * fitted_tree.predict(X)

        #Handling multi-class classification
        else:
            self.initial_predictions = numpy.zeros((X.shape[0], self.n_classes))
            current_predictions = self.initial_predictions.copy()

            for i in range(self.B):
                gradients = self.calculate_residuals(y, current_predictions)
                trees_per_class = []
                for k in range(self.n_classes):
                    tree = DecisionTree(self.max_depth, self.min_sample_leaf, self.min_information_gain, "logistic")
                    tree_data = numpy.column_stack((X, gradients[:, k]))
                    fitted_tree = tree.fit(tree_data)
                    trees_per_class.append(fitted_tree)
                    current_predictions[:, k] += self.lmd * fitted_tree.predict(X) # Update class-specific predictions

                self.tree_list.append(trees_per_class)
        #Returning the trained model
        return GradientBoostedTreeResults(self.tree_list, self.initial_predictions, self.lmd, self.loss, self.n_classes)

    #Calculating residuals for multi-class classification using the softmax function
    def calculate_residuals(self, y, current_predictions):
        p = softmax(current_predictions)
        y_one_hot = numpy.zeros((len(y), self.n_classes)) # One-hot encode labels
        y_one_hot[numpy.arange(len(y)), y.astype(int)] = 1
        return y_one_hot - p  # Residuals = true labels - predicted probabilities

class GradientBoostedTreeResults():
    def __init__(self, trees, initial_predictions, lmd, loss, n):
        self.trees = trees
        self.initial_predictions = initial_predictions
        self.lmd = lmd
        self.loss = loss
        self.n = n

    #Predicting outputs for new input data (X) using the trained model
    def predict(self, X):
        if self.loss == "square_error":
            predictions = numpy.copy(self.initial_predictions)
            for tree in self.trees:
                #Aggregating predictions from all trees
                predictions += self.lmd * tree.predict(X)
            return predictions

        #Binary classification predictions
        elif self.n == 2:
            predictions = numpy.copy(self.initial_predictions)
            for tree in self.trees:
                predictions += self.lmd * tree.predict(X)
            probabilities = sigmoid(predictions)
            #Converting probabilities to binary labels
            return (probabilities >= 0.5).astype(int)

        #Multi-class classification predictions
        else:
            predictions = numpy.copy(self.initial_predictions)
            for trees_per_class in self.trees:
                for k, tree in enumerate(trees_per_class):
                    predictions[:, k] += self.lmd * tree.predict(X)
            probabilities = softmax(predictions)
            #Returning the class with the highest probability.
            return numpy.argmax(probabilities, axis=1)