import numpy as np
from models.gradient_reg import DecisionTree, GradientBoostingRegressor

def test_decision_tree_errors():
    # Test cases for DecisionTree initialization
    try:
        tree = DecisionTree(min_samples_split=1)
    except ValueError as e:
        print("Error 1:", str(e))

    try:
        tree = DecisionTree(max_depth=-1)
    except ValueError as e:
        print("Error 2:", str(e))

    # Test cases for fit method (DecisionTree)
    tree = DecisionTree()

    try:
        tree.fit([[1, 2], [3, 4]], [1, 2])  # X and y mismatch
    except ValueError as e:
        print("Error 3:", str(e))

def test_gradient_boosting_errors():
    # Test cases for GradientBoostingRegressor initialization
    try:
        gb = GradientBoostingRegressor(n_estimators=0)
    except ValueError as e:
        print("Error 8:", str(e))

    try:
        gb = GradientBoostingRegressor(learning_rate=1.5)
    except ValueError as e:
        print("Error 9:", str(e))

    try:
        gb = GradientBoostingRegressor(max_depth=-1)
    except ValueError as e:
        print("Error 10:", str(e))

    try:
        gb = GradientBoostingRegressor(min_samples_split=1)
    except ValueError as e:
        print("Error 11:", str(e))

    # Test cases for fit method (GradientBoostingRegressor)
    gb = GradientBoostingRegressor()

    try:
        gb.fit([[1, 2], [3, 4]], [1, 2, 3])  # X and y mismatch
    except ValueError as e:
        print("Error 12:", str(e))

    try:
        gb.fit([[1, 2], [3, 4]], np.array([1, 2, 3]))  # y length mismatch
    except ValueError as e:
        print("Error 13:", str(e))

    try:
        gb.fit("not an array", [1, 2])  # X is not a numpy array
    except ValueError as e:
        print("Error 14:", str(e))

if __name__ == "__main__":
    test_decision_tree_errors()
    test_gradient_boosting_errors()

