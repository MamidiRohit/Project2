# Team members:
1. Spandana Vemula - A20527937
2. Hemanth Thathireddy - A20525346

# Gradient Boosting Regressor from Scratch

## Overview

This project implements a Gradient Boosting Regressor from scratch, following the methodology outlined in **Sections 10.9-10.10** of the book *Elements of Statistical Learning (2nd Edition)*. The model is designed to handle regression tasks by iteratively training decision trees to correct residual errors, creating a powerful ensemble capable of modeling complex, nonlinear relationships.

## Table of Contents

- [Introduction](#introduction)
- [Features of the Implementation](#features-of-the-implementation)
- [Dataset Description](#dataset-description)
- [How to Run the Project](#how-to-run-the-project)
  - [Step 1: Prerequisites](#step-1-prerequisites)
  - [Step 2: Clone or Set Up the Project](#step-2-clone-or-set-up-the-project)
  - [Step 3: Execute the Notebook](#step-3-execute-the-notebook)
- [Parameters for Tuning](#parameters-for-tuning)
- [Visualization and Analysis](#visualization-and-analysis)
- [Limitations and Challenges](#limitations-and-challenges)
- [Future Work](#future-work)

## Introduction

Gradient Boosting is a machine learning technique that builds an ensemble of weak learners (decision trees) by sequentially minimizing the residual error from previous learners. This approach is effective for regression tasks with complex, nonlinear relationships and structured/tabular data.

In this project:
- We implement the **Gradient Boosting Regressor** from scratch, using decision trees as the base learners.
- The implementation includes a `fit` method for training the model and a `predict` method for making predictions.

## Features of the Implementation

### What Does the Model Do?

The Gradient Boosting Regressor iteratively:
1. Makes an initial prediction based on the mean of the target variable.
2. Fits a decision tree to the residual errors at each iteration.
3. Updates predictions by adding a weighted contribution from each tree.

### Key Features

- **Customizable Parameters**: Number of trees, learning rate, and maximum tree depth can be tuned for optimal performance.
- **Modular Design**: The model is implemented with a clear `fit-predict` interface, making it easy to use and extend.
- **Visualization Tools**: Includes residual plots, feature importance charts, and learning curves for thorough analysis.

## Dataset Description

The project uses the **Housing Dataset**, which contains information about houses and their corresponding target variable (e.g., median house prices). The dataset includes features such as:
- **CRIM**: Per capita crime rate.
- **RM**: Average number of rooms per dwelling.
- **LSTAT**: Percentage of lower-status population.

The dataset is split into training and testing sets (80% training, 20% testing).

## How to Run the Project

### Step 1: Prerequisites

1. Install Python (3.7 or above).
2. Install the following Python libraries:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `scikit-learn`
   - `seaborn`

Use the following command to install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Step 2: Clone or Set Up the Project

1. **Download the Dataset**: If the dataset is not provided, download the `housing.csv` file.
2. **Set Up the Environment**: Place the dataset in your working directory.
3. **Open the Notebook**: Launch the project notebook in Jupyter Notebook, JupyterLab, or any Python IDE that supports `.ipynb` files.

## Step 3: Execute the Notebook

Run the cells in the notebook in order:

1. **Load the Dataset**: Read the dataset into a Pandas DataFrame.
2. **Preprocess the Data**: Split the dataset into features and target variables for training and testing.
3. **Define the Gradient Boosting Regressor Class**: Implement the model from scratch.
4. **Train the Model**: Use the training data to fit the model.
5. **Evaluate the Model**: Test the model on unseen data and calculate performance metrics.

### Example Code to Run the Model

To run the Gradient Boosting Regressor with custom parameters:

```python
# Import the Gradient Boosting Regressor
from gradient_boosting_regressor import GradientBoostingRegressor

# Initialize the model
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4)

# Train the model
gbr.fit(X_train.values, y_train.values)

# Make predictions
predictions = gbr.predict(X_test.values)

# Evaluate the performance
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
```

## Parameters for Tuning

The following parameters can be adjusted to optimize performance:

1. **`n_estimators`**: Number of decision trees in the ensemble.
2. **`learning_rate`**: Shrinks the contribution of each tree, controlling the step size during optimization.
3. **`max_depth`**: Limits the depth of each decision tree to prevent overfitting.

## Visualization and Analysis

The implementation includes several visualizations to analyze the model's performance:

1. **Predicted vs. Actual Values**
   - Scatter plot comparing predicted and actual values.

2. **Residual Plot**
   - Visualizes the residual errors to identify biases.

3. **Feature Importance**
   - Highlights the contribution of each feature to the final predictions.

4. **Learning Curve**
   - Tracks the model's error on training and testing data as more trees are added.

## Limitations and Challenges

1. **High Dimensionality**: The model may struggle with very high-dimensional or sparse datasets.
2. **Noise Sensitivity**: High noise levels in the data can amplify errors during training.
3. **Training Speed**: Training may be slow for a large number of trees or high-dimensional datasets.

## Mitigation Strategies

1. **Normalize or scale input features** to improve model convergence.
2. **Use early stopping or regularization** to prevent overfitting.

## Future Work

1. **Regularization Enhancements**: Add L1/L2 regularization to the model.
2. **Custom Loss Functions**: Extend the implementation to support custom loss functions.
3. **Early Stopping**: Implement early stopping based on validation error.

## Conclusion

The Gradient Boosting Regressor implemented in this project is a powerful tool for regression tasks. It provides flexibility, performance, and interpretability, making it suitable for a wide range of applications.

By following the instructions in this README, you can:

1. **Train the model** on any structured dataset.
2. **Tune the parameters** for optimal performance.
3. **Evaluate the results** using intuitive visualizations.

## Questions:

1. What does the model you have implemented do and when should it be used?
A: Model Interpretation: The Gradient Boosting Regressor models the ensemble of weak learners, consisting of decision trees. Each consecutive tree learns to reduce the residual errors from the previous trees. The final prediction is the weighted sum of the outputs from all trees.
Use Case: The model is used in regression problems where intricate, nonlinear relations exist between features and the target variable. It works well on structured or tabular data.

2. How did you test your model to determine if it is working reasonably correctly?
A: The model was trained on the housing dataset, where 80% of the data was used for training and the remaining 20% for testing. The performance was measured in terms of Mean Square Error (MSE) on the test set. Then, the model performance was visualized, including:
- Predicted vs. Actual scatter plot
- Residuals plot
- Feature importance analysis
- Learning curve
Results: This yields a test MSE of 5.1345, which shows reasonable predictive accuracy.

3. What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
A: Parameters exposed for tuning:
n_estimators: Number of boosting iterations (trees).
learning_rate: Controls the contribution of each tree.
max_depth: Maximum depth of individual decision trees to control overfitting.
Usage Example:
gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
max_depth=4)
gbr.fit(X_train, y_train)
predictions = gbr.predict(X_test)

4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
A: Challenges: high-dimensional datasets with sparse features noisy data, which can increase residual errors. If input features have very different scales, convergence could be slow.
Possible Solutions: standardize or normalize features to improve convergence add regularization to avoid overfitting, such as min samples per leaf.

# Contribution:

Spandana Vemula - A20527937
Did loading of dataset with the correct format, data cleaning, pre-processing, splitted the dataset into features and target variable. Trained and tested the gradient boosting regressor.

Hemanth Thathireddy - A20525346
Did Visualization and Analysis part and evaluated the performance:
1. **Predicted vs. Actual Values**
   - Scatter plot comparing predicted and actual values.
2. **Residual Plot**
   - Visualizes the residual errors to identify biases.
3. **Feature Importance**
   - Highlights the contribution of each feature to the final predictions.
4. **Learning Curve**
   - Tracks the model's error on training and testing data as more trees are added.
