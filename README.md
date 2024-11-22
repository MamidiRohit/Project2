# Project 2
- A20586642 - Nesh Rochwani
- A20602211 - Kannekanti Nikhil
- Team Name: Tree Titans

## Boosting Trees

Implement the gradient-boosting tree algorithm (with the usual fit-predict interface) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). 

Q-1. What does the model you have implemented do and when should it be used?
The implemented Gradient Boosting Tree model performs regression by combining multiple decision trees to minimize a chosen loss function (e.g., squared error, absolute error, or Huber loss). It is ideal for structured data where interpretability and performance are needed, such as predicting housing prices or financial metrics.

Q-2. How did you test your model to determine if it is working reasonably correctly?
I tested the model using the California Housing dataset by splitting it into training and testing sets, training the model, and evaluating its predictions using Mean Squared Error (MSE). Feature importance was also analyzed for interpretability.

Q-3. What parameters have you exposed to users of your implementation in order to tune performance?
Parameters include:
- n_estimators (number of trees)
- learning_rate (step size)
- max_depth (tree depth)
- loss (loss function: squared, absolute, or Huber)

Example:
gbt = GradientBoostingTree(n_estimators=100, learning_rate=0.1, max_depth=3, loss='huber')
gbt.fit(X_train, y_train)
y_pred = gbt.predict(X_test)

Q-4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
The model may struggle with noisy data or datasets with high-dimensional features due to overfitting. Adding regularization techniques, such as limiting leaf nodes or pruning, could mitigate these issues with more time. Fundamental issues arise with categorical variables, which need pre-processing.


Contribution of each Team Member:

Team Member 1 (Nesh-A20586642):
- Core Development (60% Contribution):
  - Implemented the Gradient Boosting Tree model, including tree-based regression, loss functions (squared error, absolute error, Huber loss), and the prediction mechanism.
  - Designed and added the feature importance functionality.
  - Performed testing using the California Housing dataset and evaluated model performance (e.g., MSE).

Team Member 2 (Nikhil-A20602211):
- Enhancements and Evaluation (40% Contribution):
  - Integrated hyperparameter tuning capabilities for `n_estimators`, `learning_rate`, `max_depth`, and loss functions.
  - Developed and tested visualization for feature importance using bar plots.
  - Assisted in debugging, testing edge cases, and preparing documentation for usage examples and known limitations. 

