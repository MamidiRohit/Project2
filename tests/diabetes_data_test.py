from sklearn.datasets import load_diabetes
from models.GradientBoostedTree import GradientBoostedTree
from graph_utils.utils import plot_regressiontree_results
import numpy

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

data = numpy.column_stack((X, y))
tree = GradientBoostedTree(B=50,lmd=0.1,max_depth=4, min_sample_leaf=1, min_information_gain=0.01)
tree_results = tree.fit(data)
preds = tree_results.predict(X)

plot_regressiontree_results(y,preds)

