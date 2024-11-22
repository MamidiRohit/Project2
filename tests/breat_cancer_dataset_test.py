from sklearn.datasets import load_breast_cancer
import numpy
from models.GradientBoostedTree import GradientBoostedTree
from graph_utils.utils import plot_classificationtree_results

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

data = numpy.column_stack((X, y))
tree = GradientBoostedTree(lmd=0.1,max_depth=4, min_sample_leaf=1, min_information_gain=0.01, loss="logistic")
tree_results = tree.fit(data)
preds = tree_results.predict(X)

plot_classificationtree_results(y,preds)