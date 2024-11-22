import numpy as np
from sklearn.datasets import load_iris
import numpy
import matplotlib.pyplot as plt
from models.GradientBoostedTree import GradientBoostedTree
from graph_utils.utils import plot_classificationtree_results

#Loading the Iris dataset using the `load_iris` function from sklearn.datasets
iris = load_iris()
#Extracting the feature matrix (X) containing the measurements for the Iris dataset
X = iris.data
#Extracting the target array (y) containing the class labels
y = iris.target
#Combining the (X) and (y) into a single dataset where the last column represents the target labels.
data = numpy.column_stack((X, y))

tree = GradientBoostedTree(lmd=0.1,max_depth=4, min_sample_leaf=1, min_information_gain=0.01, loss="logistic")
tree_results = tree.fit(data)

preds = tree_results.predict(X)

plot_classificationtree_results(y,preds)

