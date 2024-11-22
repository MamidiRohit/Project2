import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')

print("relative_path", relative_path)
sys.path.append(relative_path)

from gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("For some of test cases it can take a while .....")

# For Classification
# Generate some sample data with missing values for classification
np.random.seed(42)
X_clf = np.random.rand(100, 2) * 10  # 100 samples, 2 features
y_clf = (np.sin(X_clf[:, 0]) + np.cos(X_clf[:, 1]) > 1).astype(int)  # Binary target

# Introduce missing values
mask_clf = np.random.rand(*X_clf.shape) < 0.1  # 10% missing values
X_clf[mask_clf] = np.nan

# Split the data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Initialize and fit the model for classification
gb_clf = GradientBoosting(
    loss='logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    handle_missing='mean'  # Options: 'none', 'mean', 'median'
)
gb_clf.fit(X_train_clf, y_train_clf)

# Make predictions for classification
y_pred_clf = gb_clf.predict(X_test_clf)
y_proba_clf = gb_clf.predict_proba(X_test_clf)[:, 1]

# Evaluate the classification model
from sklearn.metrics import accuracy_score

acc_clf = accuracy_score(y_test_clf, y_pred_clf)
print('Classification Accuracy:', acc_clf)

gb_clf.plot_learning_curve("Professor given data")