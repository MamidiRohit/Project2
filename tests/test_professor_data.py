import csv
import os
import sys

import numpy

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')

print("relative_path", relative_path)
sys.path.append(relative_path)

from gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("For some of test cases it can take a while .....")

def test_predict():
    """
    """
    # Initialize and fit the custom model
    model = GradientBoosting(
        n_estimators=10,
        learning_rate=0.5,
        max_depth=2,
        min_samples_leaf=1,
        subsample=1.0,
        max_features=None,
        handle_missing='none'
    )

    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = numpy.array([[v for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[v for k, v in datum.items() if k == 'y'] for datum in data])
    print(X.shape)
    print(y.shape)
    model.fit(X, y)
    preds = model.predict(X)
    print(preds)
    # assert preds == 0.5
    model.plot_learning_curve("Professor given data")
    model.plot_predictions_vs_actual(preds, y)

test_predict()
