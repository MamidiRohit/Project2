# Gradient Boosting Regression

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Setup and Installation](#setup-and-installation)
4. [Model Explanation](#Setup-and-Installation)
5. [Changing Parameters via Command Line](#Changing-Parameters-via-Command-Line)
6. [Code Explanation](#code-explanation)
7. [Adjustable Parameters](#adjustable-parameters)
8. [Known Limitations](#known-limitations)
9. [Contributors](#contributors)
10. [Q&A](#qa)

---

## Project Overview
Gradient Boosting is a machine learning technique for regression and classification tasks, it combines several weak learners into strong learners where each new model is trained to minimize the loss function such as mean squared error or cross-entropy of this previous model using gradient desent. This project implements a **Gradient Boosting Regressor** from scratch, relying only on Python and NumPy. It serves as a learning tool and a practical implementation of Gradient Boosting, avoiding the use of external machine learning libraries like scikit-learn.

---

## Key Features

### Custom Implementation
- Base learners (Decision Trees) implemented from scratch without external libraries for tree-based learings tailored for regression tasks.
- Gradient Boosting logic built step by step for real-world regression problems using decision tree where each tree learns to minimize residiual erros sequentially and the predictions are updated iteratively.

### Advanced Techniques
- Supports early stopping based on validation loss.
- Stochastic Gradient Boosting via subsampling.

### Parameter Customization
#### Fully customizable hyperparameters:

1. **Model Parameters**:
   - **number of estimators**:Higher values increase capacity but may lead to overfitting without early stopping.
   - **learning rate**:Smaller values improve generalization but require more estimators.
   - **max dept**:Restricting the depth of the tree helps control complexity.
   - **criterion**: Splitting criterion for trees (default: `friedman_mse`).

2. **Regularization Parameters**:
   - **Sub sampling** : Using a fraction of the training data for each estimator introduces randomness, improving generalization. However, it may increase bias.
   - *** sample minimum split** : Defines the minimum number of samples required to split an internal node. 
   - **minimum samples per leaf**:Controls the minimum number of samples required at a leaf node. Enforcing larger leaf sizes reduces overfitting
   **alpha parameter**:Acts as a regularization term to control the complexity of trees, discouraging overly complex models and mitigating overfitting risks but might miss finer data patterns.
   -*** least_impurity reduce**: prevents unnecessary splits by ensuring meaningful reductions in impurity

3. **Stopping Criteria**:
   - **count_of_max_iteration**: Stops training if validation loss doesn’t improve.
   -*** early stopping**: improves efficiency and genralization

---

## Setup and Installation

### Prerequisites
- Python 3.x
- NumPy
- pandas

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/priyanshpsalian/Project2.git
    ```

2. Create a virtual environment
    ```bash
    python3 -m venv .env
    ```
3. Activate the environment
    ```bash
    source .env/bin/activate  # On Unix or MacOS
    .env\Scripts\activate     # On Windows
    ```
4. Get into the repo.
   ```bash
   cd Project2
   ```
5. Install the required dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```

6. Run the test file to see the results
    ```bash
    python3 -m gradientBoosting.tests.test_gradientBoostingModel
    ```

## Default File Path

- The script is configured to look for the test data in the following default location:  
`gradientboosting/tests/test_data.csv`

- If you want to use a custom CSV file, specify the file path using the `--file_path` parameter like this 
   ```bash
      python3 -m gradientBoosting.tests.test_gradientBoostingModel --file_path "gradientboosting/tests/test_data.csv"
   ```
---
## Changing Parameters via Command Line

The script allows you to modify the model's hyperparameters and test data file directly from the command line. Here's how to specify each parameter:

### Examples

1. **Use Default Parameters:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel
   ```

2. **Specify Custom Learning Rate and File Path:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel --file_path "gradientboosting/tests/test_data.csv" --rate_of_learning 0.05
   ```

3. **Change Multiple Parameters:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel --no_of_estimators 200 --max_depth 3 --sample_minimum_split 5 --criteria_for_early_stopping 0.1
   ```
---

### Adjustable Parameters:

| Parameter                  | Default Value                 | Description                                                                                   | Possible Values/Explanation                                                      |
|----------------------------|-------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `--file_path`              | `"gradientboosting/tests/test_data.csv"` | The path to the CSV file that contains the test data.                                         | Provide the path to any CSV file.                                               |
| `--no_of_estimators`       | `1000`                        | Defines how many boosting rounds will be performed to build the model.                       | Any integer value (e.g., 100, 500, 2000).                                       |
| `--rate_of_learning`       | `0.2`                         | Controls how much each tree contributes to the final prediction.                              | Any float value (e.g., 0.05, 0.1, 0.3).                                         |
| `--max_depth`              | `2`                           | Limits how deep each decision tree can grow.                                                  | Any integer value (e.g., 3, 4, 5, 6).                                           |
| `--sample_minimum_split`   | `2`                           | The smallest number of samples needed to split a node.                                        | Any integer value (e.g., 5, 10, 15).                                            |
| `--subsample`              | `1.0`                         | The percentage of the dataset used to train each tree.                                        | Any float between 0 and 1 (e.g., 0.8, 0.9).                                     |
| `--criterion`              | `"friedman_mse"`              | The metric used to decide how to split nodes during training.                                 | `friedman_mse`, `mse`, `mae` (Mean Squared Error or Mean Absolute Error).        |
| `--minimum_samples_per_leaf`| `1`                          | The minimum number of samples that must be in a leaf node.                                    | Any integer value (e.g., 2, 4).                                                 |
| `--weight_minimum_leaf_fraction` | `0.0`                   | The smallest fraction of sample weights allowed in a leaf node.                              | Any float value between 0 and 1 (e.g., 0.1, 0.2).                                |
| `--least_impurity_reduce`  | `0.0`                         | Specifies the minimum impurity reduction needed to split a node.                              | Any float value (e.g., 0.01, 0.05).                                             |
| `--random_state`           | `None`                        | A seed value to make the results reproducible.                                                | Any integer value (e.g., 42, 12345) or `None`.                                  |
| `--how_many_features`      | `None`                        | Determines how many features should be considered when looking for the best split.           | `None`, `sqrt`, `log2`, or an integer value (e.g., 5).                          |
| `--verbose`                | `0`                           | Adjusts the amount of information displayed during the training process.                      | Any integer value (e.g., 1 for less output, 10 for more detailed debugging).     |
| `--greatest_node_of_leaf`  | `None`                        | Sets the maximum number of leaf nodes allowed in a single tree.                               | Any integer value (e.g., 10, 50, 100).                                          |
| `--criteria_for_early_stopping` | `0.1`                    | Defines how much of the training data is used for validation in early stopping.               | Any float between 0 and 1 (e.g., 0.2).                                          |
| `--count_of_max_iteration` | `None`                        | Specifies how many iterations can pass without improvement before stopping early.             | Any integer value (e.g., 10, 20).                                               |
| `--tol`                    | `0.0001`                      | A small number that decides when to stop training due to lack of progress.                    | Any float value (e.g., 0.001, 0.0005).                                          |
| `--alpha_parameter`        | `0.0`                         | A parameter that helps prevent overfitting by pruning trees based on complexity.              | Any float value (e.g., 0.01, 0.1).                                              |

---


## Code Explanation 
1. **GradientBoostingRegressor**:
   ## Implements Gradient Boosting logic:
   1. **Hyperparameter Setup**: The model begins by initializing the hyperparameters and setting up attributes for the Gradient Boosting model.

   2. **Seed Initialization**: It sets a random seed using np.random.seed to ensure reproducibility of results.
   3. **Subsampling Data**: If the subsampling condition is met, a random subset of rows is selected from the dataset. If not, the entire dataset is used for training.
   4. **Model Training (fit)**: The model starts by setting an initial prediction, either from the provided init value or the mean of the target values. If early stopping is enabled, the data is split into training and validation sets. For each estimator, residuals are computed as the difference between actual and predicted values. If sub-sampling is enabled, a random subset is used for training. A decision tree is trained on the residuals, added to the model list, and the predictions are updated by adding the tree's output scaled by the learning rate. This process continues for the specified number of trees.
   5. **Prediction**: Once the model is trained, it generates predictions for new data points using the trained ensemble of trees.

 
2. **DecisionTree**:
   - ## Implements CART (Classification and Regression Trees) from scratch.
   1. **Initialization**: initializes  several hyperparameters including max_depth, sample_minimum_split, criterion, and others.If a random state is provided, the seed is set using np.random.seed to ensure reproducibility.

   2. **Mean Squared Error (MSE)**:  calculates the variance of the target values, which serves as an impurity metric for the decision tree splits.

   3. **Impurity Calculation**: It uses MSE as the impurity measure for the tree's decision-making process.

   4. **Data Splitting**: data is partitioned  into two subsets based on the threshold value of a selected feature. It returns the corresponding feature values and target values for both subsets 

   5. **Best Split Selection**: The _best_split method iterates over all features and possible threshold values to find the best split that minimizes the impurity (MSE). It randomly selects a subset of features for each split, and for each feature, it tries all possible thresholds to find the one that reduces impurity the most.

   6. **Tree Building**: the decision tree is recursively built by selecting the best split and splitting the data into left and right subsets. The recursion continues until the maximum depth is reached, the node contains only one class, or if the number of samples is too small to split further. When no valid split is found, it creates a leaf node with the mean of the target values.

   7. **Pruning**: this method simplifies the tree by merging nodes based on a cost-complexity measure. It recursively prunes the left and right subtrees, and if merging leads to a lower cost, the merge is performed. The tree is pruned if the total cost is lower than the complexity parameter alpha_parameter.

   8. **Training (fit):** The fit method trains the decision tree by building it recursively using the _build_tree method. If alpha_parameter is greater than 0, then tree is pruned using the _prune method to avoid overfitting.

   9. **Prediction**: The _predict method traverses the trained decision tree to predict the target value for a single data point. It follows the tree branches based on the feature values and returns the predicted value when a leaf node is reached.

   10. **Predicting for Multiple Data Points**: The predict method generates predictions for all input data points by calling the _predict method for each data point in the input matrix X. It returns an array of predicted values.

### Key Methods
- `fit(X, y)`: Trains the Gradient Boosting model by iteratively adding decision trees to reduce residuals.

- `predict(X)`: Makes predictions by combining the outputs of all decision trees.


## Model Evaluation Metrics for Gradient Boosting Model

1. **Coefficient of Determination (R² Score)**
- The R² Score measures the proportion of variance in the dependent variable (target) that is predictable from the independent variables (features). It quantifies how well the model's predictions align with the actual data.

- A high R² Score is indicative of the model capturing all the patterns hidden inside the data.
Significance: The metric indicates the general degree of fit of the model and shows what amount of variance in the target variable the input features explain.
2. **Mean Squared Error (MSE)**
- MSE calculates the average of the squared differences between the actual and predicted values.

- Interpretation: The metric is sensitive to large errors in prediction since the squaring of differences amplifies them.
Significance: The MSE focuses on large errors and, hence, will help to show if the model is overfitting or underfitting; it sometimes makes extreme deviations from the actual values.
3. **Mean Absolute Error (MAE)**
- While predicting a value, MAE simply calculates the average magnitude of errors between actual and predicted values without squaring them, hence treating all the errors equally.

- This is unlike MSE; hence, MAE treats all deviations equally, making it useful to understand the typical size of the prediction errors.
Significance: A low MAE reflects that, on average, the model's actual predictions are close in value. Because MAE is less sensitive to large errors, it complements MSE by offering a balanced view of model performance.
4. **Root Mean Squared Error (RMSE)**
- RMSE is a square root of MSE, and it also means the average prediction error in the same units as the target variable.

- RMSE offers a straightforward and interpretable measure for the model's prediction errors directly comparable to the scale of the target variable.
Significance: A low RMSE will indicate that the model's predictions deviate minimally from the actual values, indicating both reliability and precision.

---
## Sample Results:

| Metric                  | Value    |
|-------------------------|----------|
| **R² Score**            | 0.9876   |
| **Mean Squared Error**  | 0.0003   |
| **Mean Absolute Error** | 0.0123   |
| **Root Mean Squared Error** | 0.0175 |


--- 


## Known Limitations
### 1. **High-Dimensional Data**
- **Problem**: Increased computational complexity and risk of overfitting.
- **Solutions**:
  - Feature selection.
  - Dimensionality reduction (e.g., PCA).
  - Reducing the number of features considered per split (`how_many_features`).

### 2. **Imbalanced Datasets**
- **Problem**: Sensitivity to imbalanced target values.
- **Solutions**:
  - Adjusting sample weights.
  - Using specialized loss functions that are robust to outliers.

### 3. **Extremely Large Datasets**
- **Problem**: Excessive training time.
- **Solutions**:
  - Implementing parallel processing for tree building.
  - Increasing `subsample` to use smaller subsets of data.

### 4. **Noisy or Inconsistent Data**
- **Problem**: Overfitting to noise in the data.
- **Solutions**:
  - Aggressive pruning.
  - Adjusting `alpha_parameter` for complexity control.
  - Limiting tree depth (`max_depth`).
  - Increasing the `min_samples_leaf` parameter.

### 5. **Categorical Variables**
- **Problem**: Inefficient handling of categorical data.
- **Solutions**:
  - Automatic encoding of categorical features.
  - Implementing custom splitting criteria for categorical variables.

### 6. **Extrapolation**
- **Problem**: Poor performance on inputs outside the training data range.
- **Solutions**:
  - Ensuring training data covers the expected input range.
  - Combining with models better suited for extrapolation.

Addressing these challenges enhances the model's robustness and widens its applicability across diverse datasets.

---


## Q&A

# Detailed Analysis of the Gradient Boosting Regressor Implementation

## 1. What Does the Model Do and When Should It Be Used?


### Sequential Tree Building
- The model constructs trees sequentially, with each tree attempting to correct the errors of the ensemble of previous trees.
- This iterative process helps the model improve its predictions over time.

### Residual Learning
- At every iteration, the model calculates the residuals—differences between predicted and actual values—and trains the next tree on these residuals.
- This helps to reduce generalization error and improve the model's accuracy.

### Gradient Descent Optimization
- The model uses **gradient descent** to minimize a specified loss, which in this implementation is **Mean Squared Error (MSE)**.
- It does this to identify the best parameters for the ensemble, refining the model's performance.

### Feature and Data Subsampling
- **Feature subsampling** and **data subsampling** are part of this implementation.
- These techniques help to prevent overfitting and increase the model's ability to generalize to unseen data.

### Customizable Tree Parameters
- Users can adjust various parameters, such as:
  - Maximum depth of trees
  - Minimum samples required for splitting nodes
  - The number of features to consider for each split
- This flexibility allows fine-tuning of the model to improve performance and prevent overfitting.

### Early Stopping
- The model includes an **early stopping** mechanism based on validation loss improvement.
- This helps prevent overfitting by halting the training process when the model's performance on the validation set stops improving.

---

## When to Use the Model

### Regression Tasks
- The **Gradient Boosting Regressor** is specifically designed for **regression problems**, where the goal is to predict continuous target variables.

### Structured or Tabular Data
- It works remarkably well on **structured datasets**, often outperforming other algorithms for this type of data.

### Non-linear Relationships
- The model can discover complex **non-linear patterns** in data, making it suitable for capturing intricate relationships between features and target variables.

### Small to Medium-sized Datasets
- While the model can handle large datasets, it is particularly effective for **small to medium-sized datasets**, where other algorithms might struggle to find meaningful patterns.

### High Accuracy Requirements
- When the task demands high **predictive accuracy**, the Gradient Boosting Regressor is an excellent choice because it refines its predictions through the iterative ensemble approach.

### Feature Interactions
- The model can effectively model **strong feature interactions** due to its tree-based structure, making it ideal for datasets where interactions between features are crucial.

### Balancing Performance and Interpretability
- While the global ensemble might be hard to interpret, individual trees can still be studied to gain insights into **feature importance** and the decision-making process.
- This makes it feasible for use in applications where high performance is necessary but some level of interpretability is desired.

### Robust to Outliers
- The model is relatively **robust to outliers** due to the use of decision trees, which can be beneficial when removing outliers is not practical or desirable.

### Handling Missing Data
- **Decision trees** can handle missing data efficiently, making this model well-suited for datasets with incomplete or missing information.

### Suitable Applications:
- Regression tasks on structured or tabular data.
- Predicting continuous variables such as house prices, stock values, or sales forecasts.
- Handling non-linear relationships in data.
- Small to medium-sized datasets.
- Tasks requiring high accuracy.
- Scenarios involving significant feature interactions.

This model is particularly effective when balancing predictive performance and interpretability, as individual decision trees can provide insights into feature importance.

---

## 2. How did you test your model to determine if it is working reasonably correctly?

The Gradient Boosting Regressor was tested using a comprehensive process:

### 1. **Data Preparation**
- Geenrated the data from data generator.
- Loading data from a CSV file.
- Normalizing input features and target variables to the range [0, 1].
- Splitting the dataset into training (80%) and testing (20%) sets.

### 2. **Model Training**
- Instantiating the `GradientBoostingRegressor` with specified or default hyperparameters.
- Fitting the model to the training data.

### 3. **Prediction**
- Using the trained model to predict values on the test dataset.

### 4. **Performance Evaluation**
- Calculating various performance metrics:
  - **R² Score**: Proportion of variance explained by the model.
  - **Mean Squared Error (MSE)**: Average squared differences between predictions and actual values.
  - **Mean Absolute Error (MAE)**: Average absolute differences between predictions and actual values.
  - **Root Mean Squared Error (RMSE)**: Square root of MSE, providing an error estimate on the target variable's scale.

### 5. **Flexibility**
The implementation allows all parameters to be specified via the command line, enabling easy experimentation with different models and parameter configurations.

This robust testing ensures the model's generalization performance on unseen data is appropriately evaluated.

---

## 3. What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

The implementation provides several hyperparameters for customization and fine-tuning:
| **Parameter**                     | **Description**                                                | **Default** |
|------------------------------------|---------------------------------------------------------------|-------------|
| `no_of_estimators`                | The total number of trees that will be built in the model.     | 1000        |
| `rate_of_learning`                | The learning rate that controls the contribution of each tree. | 0.2         |
| `max_depth`                       | The maximum depth each tree is allowed to grow.               | 2           |
| `sample_minimum_split`            | The minimum number of samples required to split a node.       | 2           |
| `subsample`                       | The proportion of the dataset used for training each tree.    | 1.0         |
| `criterion`                       | The criterion used to evaluate the quality of splits.         | `friedman_mse` |
| `min_samples_leaf`                | The smallest number of samples that must be in a leaf node.   | 1           |
| `min_weight_fraction_leaf`        | The minimum fraction of sample weights that can be in a leaf node. | 0.0     |
| `min_impurity_decrease`           | The minimum reduction in impurity required to perform a split. | 0.0       |
| `random_state`                    | The seed used to ensure results are reproducible.             | None        |
| `verbose`                         | Controls the level of detail in output logs.                  | 0           |
| `stopping_criteria`               | The fraction of training data set aside for validation in early stopping. | 0.1 |
| `n_max_iteration`                 | The maximum number of iterations allowed without improvement before stopping. | None |
| `tol`                             | The minimum change in performance required to continue training. | 0.0001    |
| `alpha_parameter`                 | A parameter used for controlling the complexity of the tree pruning process. | 0.0 |

These parameters enable users to control model behavior, manage overfitting, and optimize performance for specific datasets.

### Examples

1. **Use Default Parameters:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel
   ```

2. **Specify Custom Learning Rate and File Path:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel --file_path "gradientboosting/tests/test_data.csv" --rate_of_learning 0.05
   ```

3. **Change Multiple Parameters:**
   ```bash
   python3 -m gradientBoosting.tests.test_gradientBoostingModel --no_of_estimators 200 --max_depth 3 --sample_minimum_split 5 --criteria_for_early_stopping 0.1
   ```
---

## 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

While the implementation is robust, certain input scenarios may pose challenges:

## a. High-Dimensional Data

### Problem:
High-dimensional datasets can lead to increased computational complexity and a risk of overfitting in the **Gradient Boosting Regressor**.

### Solutions:
- **Feature Selection**: Implement feature selection methods to select relevant features, such as mutual information or other correlation-based methods.
- **Dimensionality Reduction**: Apply techniques like **PCA (Principal Component Analysis)** to reduce the number of features while retaining most of the variance in the data.
- **How to Tune `how_many_features`**: The model allows adjusting the number of features considered at each split through the `how_many_features` parameter. Lowering this value helps alleviate the curse of dimensionality.

### Implementation Details:
In the `DecisionTree` class within `DecisionTree.py`, the `how_many_features` parameter is used in the `_best_split` method to randomly sample a subset of features for each split:
```python
if isinstance(self.how_many_features, str):
    if self.how_many_features == 'sqrt':
        how_many_features = int(np.sqrt(n_features))
    elif self.how_many_features == 'log2':
        how_many_features = int(np.log2(n_features))
    else:
        raise ValueError(f"Invalid value for how_many_features: {self.how_many_features}")
elif self.how_many_features is None:
    how_many_features = n_features
else:
    how_many_features = self.how_many_features

features = np.random.choice(n_features, how_many_features, replace=False)
```

---

## b. Datasets with Imbalanced Classes

### Problem:
The Gradient Boosting Regressor is sensitive to imbalanced target values in regression tasks.

### Solutions:
- **Use Weighted Samples**: Implement a weighting scheme that gives more emphasis to target values that are underrepresented.
- **Custom Loss Functions**: Develop more robust loss functions that are better suited for handling outliers or imbalanced data.

### Implementation Details:
The current implementation does not explicitly handle imbalanced datasets, but we can improve this by enabling **sample weights** in the `fit` method of `GradientBoosting.py`:
```python
def fit(self, X, y, sample_weights=None):
    # Rest of the code remains the same
    for i in range(self.no_of_estimators):
        residuals = y_train - predictions[:len(y_train)]
        X_sample, residuals_sample, weights_sample = self._subsample_data(X_train, residuals, sample_weights)
        tree = DecisionTree()
        tree.fit(X_sample, residuals_sample, sample_weight=weights_sample)
    # Rest of the method
```

---

## c. Very Large Datasets

### Problem:
Training on extremely large datasets can lead to excessive training time.

### Solutions:
- **Parallel Processing**: Implement parallel tree building using multiple CPU cores.
- **Subsampling**: Use the `subsample` parameter to train on smaller subsets of the data, reducing training time.

### Implementation Details:
Subsampling is already implemented in the `_subsample_data` method in `GradientBoosting.py`:
```python
def _subsample_data(self, X, y):
    X = np.array(X)
    if self.subsample < 1.0:
        n_samples = int(self.subsample * X.shape[0])
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        return X[indices], y[indices]
    return X, y
```

---

## d. Noise or Inconsistent Data

### Problem:
The model may overfit on noise in the data.

### Solutions:
- **Aggressive Pruning**: Use the `alpha_parameter` to control the complexity of the tree.
- **Restricting Tree Depth**: Use the `max_depth` parameter to limit the depth of trees.
- **Minimum Samples in a Leaf**: The `minimum_samples_per_leaf` parameter ensures that each leaf node contains a minimum number of samples.

### Implementation Details:
The pruning and depth restrictions are implemented in the `DecisionTree` class and used in the `_build_tree` and `_prune` methods.

---

## e. Categorical Variables

### Problem:
The current implementation may not optimally handle categorical variables.

### Solutions:
- **Automatic Encoding**: Preprocess categorical features by encoding them automatically.
- **Custom Splitting Criteria**: Develop specialized splitting criteria for categorical variables.

### Implementation Details:
The current implementation assumes numeric input. To handle categorical variables, the `_best_split` method in `DecisionTree.py` needs to be modified to handle non-numeric data types and implement appropriate splitting criteria.

---

## f. Extrapolation

### Problem:
As with all tree-based methods, **extrapolation** to ranges of data outside those used for training is problematic.

### Solutions:
- **Data Coverage**: Ensure that the training data covers the expected range of inputs.
- **Model Combination**: Combine the **Gradient Boosting Regressor** with other models that handle extrapolation better.

### Implementation Details:
Extrapolation is an intrinsic limitation of the tree-based approach, and this can only be partially addressed by changing the algorithm or combining it with other models.




## Contributors
- Priyansh Salian (A20585026 psalian1@hawk.iit.edu)
- Shruti Ramchandra Patil (A20564354 spatil80@hawk.iit.edu)
- Pavitra Sai Vegiraju (A20525304 pvegiraju@hawk.iit.edu)
- Mithila Reddy (A20542879 Msingireddy@hawk.iit.edu)



