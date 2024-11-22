import numpy

class DecisionTree():
    #Initializing the Decision Tree with hyperparameters
    """
    max_depth: Maximum depth of the tree
    min_sample_leaf: Minimum samples required in a leaf node
    min_information_gain: Minimum information gain for a split to be considered
    loss: Loss function to optimize. Default is square_error
    """
    def __init__(self, max_depth=5, min_sample_leaf=5, min_information_gain=0.0, loss="logistic") -> None:
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.min_information_gain = min_information_gain
        self.loss = loss

    #Computing the entropy value of class probabilities (class_prob)
    def entropy(self, class_prob):
        return numpy.sum([-p * numpy.log(p) for p in class_prob if p > 0])

    #Calculating entropy for the given dataset's labels
    def entropy_data(self, labels):
        unique, counts = numpy.unique(labels, return_counts=True)
        probs_each_label = counts / labels.shape[0]
        return self.entropy(probs_each_label)

    #Computing the weighted average entropy of partitions(subsets)
    def partition_entropy(self, subsets):
        count = numpy.sum([subset.shape[0] for subset in subsets])
        return numpy.sum([self.entropy_data(subset[:, -1]) * subset.shape[0] for subset in subsets]) / count

    #Spliting the dataset(data) based on a feature (f_id) and its value (f_value)
    def feature_split(self, data, f_id, f_value):
        mask = data[:, f_id] < f_value
        g1 = data[mask]
        g2 = data[~mask]
        return g1, g2
    #Finding the best feature and value & best and minimum partition to split the data on
    def get_best_split(self, data):
        min_partition_entropy = float('inf')
        min_entropy_feature = None
        min_entropy_value = None
        min_g1 = None
        min_g2 = None

        for id in range(data.shape[1] - 1):
            unique_feature_values = numpy.unique(data[:, id])
            for value in unique_feature_values:
                g1, g2 = self.feature_split(data, id, value)
                if len(g1) < self.min_sample_leaf or len(g2) < self.min_sample_leaf:
                    continue
                partition_entropy = self.partition_entropy([g1, g2])
                if partition_entropy < min_partition_entropy:
                    min_partition_entropy = partition_entropy
                    min_entropy_feature = id
                    min_entropy_value = value
                    min_g1 = g1
                    min_g2 = g2

        return min_g1, min_g2, min_entropy_feature, min_entropy_value, min_partition_entropy

    #Creating a leaf node for the tree
    def leaf_node(self, data, labels):
        if self.loss == "square_error":
            prediction = numpy.mean(labels)
        else:
            prediction = numpy.mean(labels)
        return TreeNode(data, None, None, prediction, 0)

    #Recursively building the decision tree and returns the root of the tree
    def build_tree(self, data, current_depth):
        labels = data[:, -1]
        if (current_depth >= self.max_depth) or (len(numpy.unique(labels)) == 1):
            return self.leaf_node(data, labels)

        g1, g2, feature, value, information_gain = self.get_best_split(data)

        if feature is None or information_gain < self.min_information_gain:
            return self.leaf_node(data, labels)

        left = self.build_tree(g1, current_depth + 1)
        right = self.build_tree(g2, current_depth + 1)

        prediction = numpy.mean(labels)
        new_node = TreeNode(data=data, feature_id=feature, feature_value=value, prediction=prediction, information_gain=information_gain)
        new_node.left = left
        new_node.right = right
        return new_node

    #Fitting the decision tree model to the data
    def fit(self, data):
        tree_root = self.build_tree(data, 0)
        return DecisionTreeResults(tree_root=tree_root)

class TreeNode():
    #Representing a single node in the decision tree
    ''' data: Data at the node
      feature_id: Feature index used for splitting
      feature_value: Feature value to split on
      prediction: Prediction value at the node
      information_gain: Information gain for the split
    '''
    def __init__(self, data, feature_id, feature_value, prediction, information_gain) -> None:
        self.data = data
        self.feature_id = feature_id
        self.feature_value = feature_value
        self.prediction = prediction
        self.information_gain = information_gain
        self.left = None
        self.right = None

class DecisionTreeResults():
    #Wrapping the trained decision tree for making predictions
    def __init__(self, tree_root) -> None:
        self.tree_root = tree_root

    #Predicting the output for a single data row
    def predict_single_row(self, node, row):
        if node.feature_id is None:
            return node.prediction
        if row[node.feature_id] < node.feature_value:
            return self.predict_single_row(node.left, row)
        else:
            return self.predict_single_row(node.right, row)

    #Predicting the outputs for the entire dataset
    def predict(self, data):
        result = []
        for row in data:
            result.append(self.predict_single_row(self.tree_root, row))
        return numpy.array(result)