from sklearn.datasets import load_wine
from models.GradientBoostedTree import GradientBoostedTree
from graph_utils.utils import plot_classificationtree_results
import numpy

wine = load_wine()
X = wine.data
y = wine.target

data = numpy.column_stack((X, y))
tree = GradientBoostedTree(lmd=0.1,max_depth=4, min_sample_leaf=1, min_information_gain=0.01, loss="logistic")
tree_results = tree.fit(data)
preds = tree_results.predict(X)

plot_classificationtree_results(y,preds)