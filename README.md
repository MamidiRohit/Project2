# Project 2
Student: Yuxuan Qian (A20484572)

## Boosting Trees
* __What does the model you have implemented do and when should it be used?__<br />
The gradient boosting tree I had implemented is one of the machine learning methods that can predict the regression. Since the gradient boosting algorithm trains data by combining several weak learners, I first created the `DcisionTreeRegressor` class which is applied to predict the regression data by decision tree to support a learner that the gradient boosting algorithm can integrate. Thus, the model will be utilized if we want to predict the data for linear regression.
  
* __How did you test your model to determine if it is working reasonably correctly?__<br />
I tested my model with 4 test cases. The first three cases are randomly generated regression data, and the last is the data about US housing prices. All data were first split into 80% training dataset and 20% test dataset. After importing the training data into the model, we can predict the by inputting x-testing data into the trained model. Finally, I measured the performance of the model by r square and the plot of the actual vs the prediction. The r square in all tests is more than 0.85, and the plot is roughly a line through the origin. Those show the model predicted well.
  
* __What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)__<br />
The model provides 4 parameters: `n_estimators` (number of the estimator), `rate` (learning rate), `max_depth` (maximum depth of the tree), and `min_samples_split` (the splitting require of minimum samples). Their default values are respectively 100, 0.1, 3, and 2. Users can change the parameter values to get more accurate performance or faster training.


* __Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?__<br />
