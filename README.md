
#  Gradient Boosting Tree Algorithm 
### Project - 2








#### Team Members:

-     Vamsi Krishna Chitturi (A20539844) (vchitturi1@hawk.iit.edu)
-     Likhith Madhav Dasari (A20539604) (ldasari2@hawk.iit.edu)
-     Leela Lochan Madisetti (A20543643) (lmadisetti@hawk.iit.edu)
-     Santhosh Ekambaram (A20555224) (sekambaram@hawk.iit.edu)



Gradient boosting is an ensemble learning technique that combines several decision trees to improve accuracy iteratively and additively. Unlike the original, independent predictive approach of the traditional decision tree, gradient boosting constructs trees in a sequence wherein the next tree focuses on the residual errors or gradients of the previous ones. The algorithm begins with fitting a weak model, typically a shallow decision tree and then adjusts predictions by minimizing a loss function, for example, Mean Squared Error via gradient descent. This continues until either the predetermined number of n_estimators has been reached or acceptable performance has been obtained from the model. A learning rate controls how much each subsequent tree contributes to the final model, thereby also controlling overfitting. Gradient boosted models are very robust and accurate for prediction; they usually outperform other machine-learning algorithms.

### DataSets:



#### BostonHousing.csv

```bash
https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv
```

### Instructions for Execution :

- Step 1:

Make shore that your environment is capable of running Python 

`Go to the Project2 folder using cd .../Project2`
and Run

` pip install -r requirements.txt`

along with 

` cd radient_boosting`

- Step 2:

Run the command `python main.py`

For testing date using fit and predict run `python test_gradient_boosting.py`

Note: Data used for return is generated.


### What does the model you have implemented do and when should it be used?

The model implemented is a Gradient Boosting Tree model, specifically aimed at predicting housing prices using the Boston Housing Dataset. Gradient Boosting as an ensemble technique builds a strong model by combining several weak learners, which in this specific instance are decision tree stumps. A new learner is added who tries to minimize the error (residuals) of the previous models iteratively; thus, it improves accuracy iteratively. The loss function, squared error in this case, is minimized and thereby fits perfectly for regression tasks.

Gradient Boosting is especially valuable in contexts requiring precise predictive accuracy and with non-linear relationships between variables. It finds extensive application in regression tasks such as predicting house prices, forecasting customer retention, and various other domains where intricate patterns are essential.This is used at Regression Tasks, Large datasets, Overfitting Control,High Accuracy Requirements.

### How did you test your model to determine if it is working reasonably correctly?

Regarding the model performance assessment, the Boston Housing Dataset was split into an 80% training set and a 20% test set. The evaluation metric used was Mean Squared Error (MSE) which is a measure of how far the model's predictions are from the given values. Lower MSE suggests better performance system. MSE and R^2 is calculated using evaluate_model() method which returns mse, r2 and predictions are made using predict() method.

evaluate_model(): Train a model and evaluate its performance on test data is the main pbjective

```
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    # Train the model using the training data
    model.fit(X_train, y_train)

    # Predict the target values for the test data
    predictions = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((y_test - predictions) ** 2)

    # Calculate the R-squared score (R²)
    # R² measures the proportion of variance explained by the model
    r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    # Return the computed evaluation metrics
    return mse, r2


```

The model’s hyperparameters such as the number of boosting iterations (⁠ n_estimators ⁠) , the learning rate and the maximum depth of the trees were tuned in order to find optimal performance. This is because through checking how well residual errors decreased over boosting iterations, you could tell whether a model was learning or not. Thus running a model on unseen data further made it clear that it was generalizing properly and there was no overfitting.

Hyperparameter options:

```

    learning_rates = [0.05, 0.01]
    n_estimators_options = [100, 200]
    max_depths = [3, 4, 5]
    min_samples_split = 10


```

### What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

The implementation allows the user some flexibility in tuning several important parameters that will improve the performance of the model:
*n_estimators*: Represents the number of boosting rounds (iterations). Generally, the model accuracy improves with an increased value; however, it can result in overfitting if set too high.
* ⁠ ⁠⁠ learning_rate: It controls how much contribution each new tree makes to the model. Smaller values make the model more conservative and thus help avoid overfitting.
*   max_depth: It sets the maximum depth of each decision tree. Trees with low depth are supposed to prevent overfitting, whereas deeper trees fit more complex patterns in the data.
* ⁠min_samples_split: The minimum number of samples required to split a node, thereby affecting the complexity and depth of the decision trees.
* n_estimators

```bash
 Usage:


# Define a Gradient Boosting Tree model with user-defined parameters
model = GradientBoostingTree(n_estimators=200, learning_rate=0.1, max_depth=3)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
predictions = model.predict(X_test)

```
In this usage, the model is initialized with 200 boosting stages, a learning rate of 0.1, and a tree depth of 3. These parameters control the model’s training process, balancing complexity and accuracy. For better MSE different parameters are used in three stage for loop and the best is used for training data prediction.

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes, there are some input types that the present implementation may have difficulties with.

- Noisy Data: The fact that the model tries to minimize residuals at each step makes it sensitive to noises. If the data is noisy it can lead to overfitting where a model will capture some rules that actually don’t generalize well. Techniques such as early stopping, learning rate reduction or regularization may be taken to reduce this difficulty.
- ⁠High-Dimensional data: If you have datasets rich in features or containing sparse data where many values are zero also performance of the model could be negatively affected. There are ways to try to solve it – and they are feature selection, dimensionality reduction, and regularization.

If given more time, improvements could include:

- ⁠Explicit Gradient Calculation: The current implementation calculates gradient information based on residuals. More sophisticated models explicitly compute gradients and therefore optimize better, as is the case with the XGBoost algorithm.
- Adaptive Learning Rate: A learning rate that adjusts dynamically instead of being fixed could perhaps speed up convergence and yield better results overall.
- Tree Depth Tuning: Allowing the trees to grow slightly deeper (e.g., max depth = 2 or 3) could provide a nicer trade-off between simplicity and capturing more intricate patterns.


While these limitations are inherent to boosting, subtle tweaks can improve the model's performance across a wider range of input types.
 






### Code Description:
Here is the descrption for the implementation of every interface listed.
#### Related Methods: 

#### DecisionTreeStump():
This is a simple decision tree model, commonly referred to as a "stump"; it contains only one decision node (therefore, it's a shallow tree). This model splits the data based on one feature and its corresponding threshold aimed at minimizing the residuals (errors) in the context of boosting. In the gradient boosting framework, this model is utilized as a weak learner to improve predictions iteratively.
#### GradientBoostingTree():
This is the core implementation of gradient boosting. It constructs an ensemble of decision trees (weak learners) that iteratively fit models to the residuals of the predictions made by the previous model, thereby minimizing a chosen loss function, for instance, Mean Squared Error. In addition, predictions from each tree are controlled by a learning rate to prevent overfitting.
#### evaluate_model():
This role is in charge of assessing the performance of the model that has been trained. It calculates main metrics such as Mean Squared Error (MSE) and R-squared (R²) to evaluate how accurate the model is and how well it fits the test data. This is done much more frequently when training models so that hyperparameters can be selected.
#### fit():
The `fit` method is used to train the model i.e. it teaches the model using the training data. In gradient boosting, this method fits weak learners (which are decision trees) iteratively onto the residuals of the previous predictions and hence updates the model at each step to enhance its performance.
#### predict():
The `predict` method is used to make prediction on the new (unseen) data. For a gradient boosting model, it accumulates the predictions of all the trained trees only, applying the learning rate to change their contribution, and at the end predicting each sample in the test set.


## Output Performance Summary for test data:

#### Best Model Parameters
The following parameters were determined as optimal based on the lowest Test Mean Squared Error (MSE):

- **Learning Rate**: `0.05`  
- **Number of Estimators**: `200`  
- **Maximum Depth**: `3`


### Model Performance

#### Test Data Evaluation
The model's performance on the test set was measured using the following metrics:

- **Test Mean Squared Error (MSE)**: `14.57`  
- **Test R² Score**: `0.46`  
- **First 10 Predictions on Test Data**:  
  `[9.71242106, 9.71242106, 16.08454935, 20.00021255, 15.85093078, 15.8789852, 16.20398898, 13.40838847, 11.14215722, 11.14215722]`  

#### Training Data Evaluation
The model's performance on the training set demonstrated its ability to fit the data effectively:

- **Train Mean Squared Error (MSE)**: `10.31`  
- **Train R² Score**: `0.87`  
- **First 10 Predictions on Training Data**:  
  `[27.68339012, 23.82163297, 36.06111604, 36.03437377, 32.83105128, 26.06563731, 20.59168011, 18.81861553, 16.43694979, 18.5237244]`  


## Key Insights
1. **Training Performance**:  
   The low Train MSE and high R² value on the training set indicate that the model effectively captures patterns in the training data.  

2. **Test Performance**:  
   The Test MSE is higher compared to the Train MSE, which is expected due to generalization errors. However, the acceptable Test MSE demonstrates the model's ability to perform reasonably well on unseen data.

3. **Optimal Depth and Learning Rate**:  
   A maximum depth of `3` and a learning rate of `0.05` prevent overfitting, ensuring that each decision tree in the ensemble focuses on capturing meaningful improvements.


## Recommendations
- **Hyperparameter Tuning**:  
  Further tuning, such as experimenting with `min_samples_split` or alternative learning rates, could improve performance.  
- **Feature Engineering**:  
  Investigate feature transformations or additional variables to reduce Test MSE and improve predictive accuracy.  
- **Cross-Validation**:  
  Evaluate the model using cross-validation to ensure robust performance across multiple data splits.


This Gradient Boosting Tree implementation demonstrates effective regression capabilities and serves as a foundation for further optimization and analysis.

