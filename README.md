# Project 2

## Gradient Boosted Trees (GBT) Implementation and Application

## Team Members:

- **Aakash Shivanandappa Gowda**  
  A20548984
- **Dhyan Vasudeva Gowda**  
  A20592874
- **Hongi Jiang**  
  A20506636
- **Purnesh Shivarudrappa Vidhyadhara**  
  A20552125

## 1. Overview

This project implements a **Gradient Boosted Trees (GBT)** model from scratch for both classification and regression tasks. The model is designed to iteratively improve its predictions by correcting the residuals of previous iterations using regression trees. It provides a practical demonstration of

## 2. Key Components of the Project

### 1. Gradient Boosted Trees (GBT) Implementation

The core functionality of the project lies in the custom implementation of Gradient Boosted Trees:

- **`GBT` Class**:
  - Implements the Gradient Boosting algorithm for regression tasks.
  - Supports multiple decision trees as weak learners to iteratively improve the prediction accuracy.
  - Uses the residuals of the previous tree to train the next tree.
  - **Parameters**:
    - `num_estimators`: Number of boosting stages.
    - `max_depth`: Maximum depth of each regression tree.
    - `min_split`: Minimum samples required to split an internal node.
    - `learning_rate`: Step size for updating predictions.
    - `criterion`: Error metric used to evaluate the split (`mse` for regression tasks).

- **`RegressionTree` Class**:
  - Implements individual regression trees.
  - Recursively splits the data to minimize the residual error at each node.
  - Supports finding the best split based on minimizing mean squared error (MSE).
  - Leaf nodes return the average target value for regression.

### 3. Dataset Handling

The project includes handling two datasets:

- **Iris Dataset (Classification)**:
  - Built-in dataset from `sklearn.datasets`.
  - Used to classify flower species.
  - Model predictions are rounded to the nearest class labels.
  - Classification accuracy is calculated using `accuracy_score`.

- **Concrete Data Dataset (Regression)**:
  - Provided via an Excel file (`Concrete_Data.xls`).
  - Used to predict compressive strength based on features such as cement, water, and age.
  - Evaluation metrics include **Root Mean Squared Error (RMSE)** and **R² Score**.

### 4. Features of the Project

- **Training and Evaluation**:
  - Supports training GBT models on both classification and regression tasks.
  - Evaluates the model using relevant metrics (**accuracy**, **RMSE**, and **R²**).
  - Automatically splits data into training and testing sets using `train_test_split`.

- **Model Persistence**:
  - Models are saved as `.pkl` files using Python’s `pickle` module.
  - This allows for loading pre-trained models for future use without re-training.

- **Visualization**:
  - Uses `matplotlib` to visualize:
    - **Predictions vs. True Values**.
  - Helps assess model performance visually.

### 5. Main Functionalities

- **`train_and_save_model()`**:
  - Trains a GBT model on the Iris dataset.
  - Evaluates the model on the test set and saves the model as `gbt_iris_model.pkl`.
  - Generates a plot comparing predicted and true values.

- **`load_and_plot_model()`**:
  - Loads a pre-trained GBT model from disk.
  - Evaluates it on the test set and plots predictions against true values.

- **`train_concrete_model()`**:
  - Trains a GBT model on the Concrete dataset.
  - Evaluates the model using **RMSE** and **R²**.
  - Saves the model and generates a plot comparing predictions and true values.

- **Interactive Command-Line Menu**:
  - Provides users with the following options:
    1. Train and save the Iris model.
    2. Load the saved Iris model and visualize predictions.
    3. Train the default Concrete dataset model.
    4. Train a custom dataset model provided by the user.
    5. Quit the application.

### 6. Evaluation Metrics

- **Classification Tasks (Iris Dataset)**:
  - **Accuracy Score**: Measures the proportion of correctly classified samples.

- **Regression Tasks (Concrete Data)**:
  - **Root Mean Squared Error (RMSE)**: Measures the standard deviation of prediction errors.
  - **R² Score**: Indicates how well the model explains the variance in the target variable.

## 1. Boosting Trees

## Q1.What does the model you have implemented do and when should it be used?                                  #################




## Q2.How did you test your model to determine if it is working reasonably correctly?


## Q3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)



## Q4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

