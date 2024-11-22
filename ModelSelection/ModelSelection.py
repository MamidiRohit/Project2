#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class ModelSelection:
    def __init__(self, model, loss_function):
        """
        Initialize the model selector with a given model and loss function.

        Parameters:
        - model: A class with `fit` and `predict` methods.
        - loss_function: A callable that takes (y_true, y_pred) and returns a scalar loss.
        """
        self.model = model
        self.loss_function = loss_function

    def k_fold_cross_validation(self, X, y, k=5):
        """
        Perform k-fold cross-validation.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - k: Number of folds (default is 5).

        Returns:
        - mean_loss: The average loss across all folds.
        """
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k
        losses = []

        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            loss = self.loss_function(y_test, y_pred)
            losses.append(loss)

        mean_loss = np.mean(losses)
        return mean_loss

    def bootstrap(self, X, y, B=100):
        """
        Perform bootstrap resampling to estimate prediction error.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - B: Number of bootstrap samples (default is 100).

        Returns:
        - mean_loss: The average loss across all bootstrap samples.
        """
        n = len(y)
        losses = []

        for _ in range(B):
            bootstrap_indices = np.random.choice(np.arange(n), size=n, replace=True)
            oob_indices = np.setdiff1d(np.arange(n), bootstrap_indices)

            if len(oob_indices) == 0:  # Skip iteration if no OOB samples
                continue

            X_train, X_test = X[bootstrap_indices], X[oob_indices]
            y_train, y_test = y[bootstrap_indices], y[oob_indices]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            loss = self.loss_function(y_test, y_pred)
            losses.append(loss)

        mean_loss = np.mean(losses)
        return mean_loss

    def evaluate_model(self, X, y, method='k_fold', **kwargs):
        """
        Evaluate the model using the specified method.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - method: 'k_fold' or 'bootstrap'.
        - kwargs: Additional parameters for the evaluation method.

        Returns:
        - loss: The evaluation loss.
        """
        if method == 'k_fold':
            return self.k_fold_cross_validation(X, y, **kwargs)
        elif method == 'bootstrap':
            return self.bootstrap(X, y, **kwargs)
        else:
            raise ValueError("Unsupported method. Choose 'k_fold' or 'bootstrap'.")


# In[3]:


#Dataset3
# Load dataset
import pandas as pd
file_path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
data = pd.read_csv(file_path, names=columns)

# Features and Target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Train Perceptron (Binary Classification)
class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            for i in range(len(y)):
                prediction = 1 if X[i] @ self.weights > 0 else 0
                self.weights += self.lr * (y[i] - prediction) * X[i]

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        return (X @ self.weights > 0).astype(int)

model = Perceptron(lr=0.1, epochs=100)
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
print("First 10 Predictions:", predictions[:10])


# In[ ]:




