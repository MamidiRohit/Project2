# README.md

## Team Members

- **Darshan Sasidharan Nair** : dsasidharannair@hawk.iit.edu
- **Ishaan Goel** : igoel@hawk.iit.edu
- **Ayesha Saif** : asaif@hawk.iit.edu
- **Ramya Arumugam** : rarumugam@hawk.iit.edu

## Gradient Boosting Tree Implementation
This repository provides an implementation of the Gradient Boosting Tree (GBT) algorithm. The implementation follows a modular structure with a fit-predict interface, supporting regression and classification tasks.

---

## 1. Model Description

### What does the model do?
Gradient Boosting Trees are generally used for regression and classification problems. They work by combining predictions from multiple weak learners (in this case, decision trees) to form a strong learner. By iteratively improving and optimizing upon a loss function, the GBTs build models that correct the errors and reduce the loss of previous iterations.

### When should it be used?
- For tabular datasets with complex non-linear relationships.
- When interpretability and feature importance are needed.
- In problems where hyperparameter tuning is feasible to improve performance.

---

## 2. Testing the Model

### How was the model tested?
- **Unit Tests:** The DecisionTree class and its methods were tested on datasets to ensure correct splits, entropy calculations, and tree-building logic.
- **Classification Testing:** Tested on datasets like Iris, Wine, and Breast Cancer for binary classification tasks.
- **Regression Testing:** Tested using datasets like Diabetes to predict continuous outcomes.

---

## 3. User-Exposed Parameters

### Parameters for tuning:
| Parameter | Description | Default Value |
|-|-|-|
| `B` | Number of boosting iterations | 100 |
| `lmd` | Learning rate for updates | 0.1 |
| `max_depth` | Maximum depth of each decision tree | 3 |
| `min_sample_leaf` | Minimum samples required to form a leaf | 5 |
| `min_information_gain`| Minimum information gain for a split | 0.0 |
| `loss` | Loss function (`"logistic"` for classification, `"square_error"` for regression) | `"logistic"` |

The last 4 parameters are passed into each of the `B` instances of the Decision Tree.

### Usage Example:
```python
# Sample Usage
import numpy as np

# Dummy dataset: Features (X), Labels (y)
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, size=(100,))

# Combine features and labels for input
data = np.column_stack((X, y))

# Create Gradient Boosting Model
model = GradientBoostedTree(B=50, lmd=0.05, max_depth=4)

# Fit the model
fitted_model = model.fit(data)

# Make predictions
predictions = fitted_model.predict(X)
print(predictions)
```

---

## 4. Model Limitations

### Known challenges:
- The algorithm's computational efficiency decreases with high-dimensional feature spaces, impacting the speed of decision boundary calculations.
- Risk of overfitting exists when using large numbers of boosting iterations (`B` parameter), requiring careful hyperparameter selection.
- Model accuracy can be compromised when working with class-imbalanced data unless specific countermeasures are employed.

### Potential improvements:
- Integration of regularization mechanisms to enhance model generalization
- Development of validation-based early stopping criteria
- Expansion of the loss function library to better handle various classification scenarios
- The classifier works really well for classification tasks, but not as well for regression tasks, so it is recommended for classification (which is what the assignment requires and what the professor mentioned)