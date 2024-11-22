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


# In[2]:


# Dataset 2: Multi-Class Logistic Regression
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Digits dataset
digits = load_digits()
X, y = digits.data, digits.target  # Features and target

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multi-class Logistic Regression with Softmax
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.coefficients = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, num_classes):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        self.coefficients = np.random.randn(num_classes, X.shape[1]) * 0.01  # Random initialization

        for epoch in range(self.epochs):
            logits = X @ self.coefficients.T  # Linear combination
            probabilities = self._softmax(logits)  # Apply softmax activation
            
            # Create one-hot encoded target matrix
            y_one_hot = np.zeros((y.size, num_classes))
            y_one_hot[np.arange(y.size), y] = 1
            
            # Compute gradient
            gradient = X.T @ (probabilities - y_one_hot) / len(y)
            self.coefficients -= self.lr * gradient.T  # Update coefficients

    def predict(self, X):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        logits = X @ self.coefficients.T
        probabilities = self._softmax(logits)
        return np.argmax(probabilities, axis=1)

# Initialize and train the model
model = LogisticRegression(lr=0.01, epochs=5000)
num_classes = len(np.unique(y_train))
model.fit(X_train, y_train, num_classes)

# Evaluate
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
print("First 10 Predictions:", predictions[:10])
print("Actual Labels:        ", y_test[:10])


# In[ ]:




