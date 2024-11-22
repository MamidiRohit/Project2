### Team Members

1. Satya Mani Srujan Dommeti  
2. Arjun Singh  
3. Akshitha Reddy Kuchipatla  
4. Vamsi Krishna  



# Model Implementation
This project includes the implementation of both linear regression and ridge regression. The models are designed to run concurrently, allowing the selection of the one that performs best based on the given dataset. To ensure robust performance assessment, k-fold cross-validation and bootstrapping techniques were applied. Additionally, the Akaike Information Criterion (AIC) was used to evaluate the quality and effectiveness of each model.

### K-fold Cross-Validation
K-fold cross-validation is a model validation technique used to assess the performance of a machine learning model. The dataset is split into K equally sized folds (subsets). For each fold, the model is trained using the remaining K-1 folds and tested on the current fold. This process is repeated K times, each time with a different fold serving as the test set. The final performance metric (e.g., Mean Squared Error) is averaged over all K iterations, providing a more reliable estimate of the model's generalization ability. The number of folds (K) can be adjusted based on the dataset size and computational resources, with the default set to 6 in our implementation.

### Bootstraping
Bootstrapping is a resampling technique used to estimate the performance of a machine learning model. It involves generating multiple random subsets (with replacement) from the training dataset, each of the same size as the original dataset. For each resampled subset, a model is trained, and its performance is evaluated on the test set. This process is repeated for a specified number of iterations (n_iterations), and the performance metrics (such as Mean Squared Error) are averaged to provide a more robust estimate of the model’s accuracy. The number of iterations can be adjusted based on the dataset size and computational resources, with the default set to 100 in our implementation. Bootstrapping helps assess the stability of the model by testing it on multiple resampled subsets.

### Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
In simple cases like linear regression, the cross-validation, bootstrapping, and AIC model selectors generally align, as all three methods aim to assess model performance in terms of its ability to fit the data while avoiding overfitting.

- **Cross-Validation**: By splitting the data into multiple folds and evaluating the model's mean squared error (MSE) on the validation set, k-fold cross-validation provides a robust estimate of the model's performance across different data splits.

- **Bootstrapping**: This method repeatedly samples the training data with replacement, fits the model, and evaluates its performance (e.g., MSE). It captures variability in model performance due to data sampling.

- **AIC**: AIC uses the MSE and penalizes model complexity based on the number of parameters. For linear regression, which typically has a straightforward relationship between features and output, AIC often agrees with the other methods because the MSE forms the foundation for its calculations.
  
- Both k-fold cross-validation and bootstrapping rely on MSE for evaluation, similar to AIC. Hence, in simpler models like linear regression, these approaches are likely to identify the same optimal model because they evaluate similar metrics under slightly different conditions.
- The alignment may diverge in more complex scenarios, such as models with high regularization (e.g., ridge regression) or datasets with noise, where AIC’s penalty on model complexity could lead to different selections compared to resampling-based techniques.

### In what cases might the methods you've written fail or give incorrect or undesirable results?
Bootstrapping:

May lead to overfitting by emphasizing specific patterns in the data, especially when there are repeated patterns due to resampling with replacement.
Assumes the data is representative of the population, which can cause biased estimates if the data is highly skewed or the sample size is too small.
K-fold Cross-Validation:

Can be computationally expensive for large datasets since it requires retraining the model for each fold, slowing down the process.
Might give biased results when dealing with imbalanced datasets or non-i.i.d. data (like time series), as improper data partitioning may overestimate the model’s performance.
AIC (Akaike Information Criterion):

Tends to penalize models with many predictors too much, especially in the presence of multicollinearity, potentially leading to overly simplistic models that underfit the data.
Assumes the model is correctly specified. If the model assumptions (e.g., linear relationships) are incorrect, AIC may suggest a poor-fitting model.

### What could you implement given more time to mitigate these cases or help users of your methods?
Given more time, there are several ways to improve the robustness and flexibility of the methods for linear regression, ridge regression, and model evaluation. Below are some considerations and improvements that could be made:

Improving Robustness for Linear and Ridge Regression Models:

Regularization Techniques: For Ridge Regression, we could implement adaptive regularization methods (such as randomForest Regressor) to improve performance, especially in the presence of multicollinearity or when the dataset has many features. This would allow for a more flexible model that adapts to various kinds of data and better generalization.
Handling Outliers: We could implement outlier detection and handling mechanisms (such as robust regression techniques or filtering out extreme values) to prevent the model from overfitting to noisy or extreme data points.
Enhancing Data Cleaning and Preprocessing:

Handling Missing Data: Instead of removing rows with missing values, we could implement methods for imputing missing data, such as filling with the mean, median, or mode of the column for numerical data, or using the most frequent category for categorical data. This would help retain valuable information and prevent the loss of data due to missing values, which is a common issue in real-world datasets.
Feature Scaling and Transformation: For better model convergence and accuracy, especially in models like Ridge Regression, feature scaling (e.g., standardization or normalization) could be applied. This would ensure that all features are on the same scale, preventing any feature from dominating the learning process due to larger values.
Customizable Parameters:

Making Parameters More Generic: The value of k in K-fold cross-validation and the number of iterations (n_iterations) in bootstrapping could be made more flexible by allowing users to pass these as arguments when calling the respective functions. This would make the methods more adaptable and user-friendly, especially for different use cases or datasets.
Dynamic Model Evaluation: We could enhance the evaluate_model method to include additional performance metrics (e.g., R-squared, Adjusted R-squared) to provide a more complete evaluation of model performance. This would give users more insight into how well the model is fitting the data, beyond just MSE and AIC.
Data Type Handling:

Categorical Data Handling: As we are currently not handling categorical variables, introducing preprocessing techniques like one-hot encoding or label encoding for categorical data would allow for a more comprehensive model that works with both numerical and categorical data. This is especially important for datasets with mixed data types.



### What parameters have you exposed to your users in order to use your model selectors.

The model selectors provide several important parameters that allow users to customize the model evaluation and selection process. These parameters include the number of folds (k) for cross-validation, which defaults to 5 but can be adjusted depending on the dataset size and computational constraints. For bootstrapping, users can set the number of iterations (n_iterations, with a default of 100) to balance estimation accuracy with computational efficiency. In the Ridge Regression model, the regularization strength can be tuned using the alpha parameter. Additionally, the test_size parameter (defaulting to 0.2 or 20% of the data) lets users control the proportion of data allocated for testing during data splitting. These customizable parameters offer flexibility, enabling users to tailor the model selection process to their specific requirements while ensuring proper cross-validation practices as shown in the image.

# Intructions on how to use the model.py and test.py.
Here's a paraphrase of the instructions for model.py and test.py:

**model.py:**

* Keep it simple! Make sure the data file (like "winequality-red.csv") and the test script (test.py) are in the same location.
* Double-check that you have all the necessary libraries installed for the code to work.

**test.py:**

* This script deals with loading the data. When you use the script to import the data, pay attention to the `delimiter` parameter (separates values in the file).
* The quality score (or whatever your target value is called) should be the last column in the data. Update the lines:

```python
X = df.drop('FEATURE_NAME', axis=1).values
y = df['FEATURE_NAME'].values
```

* Replace `'FEATURE_NAME'` with the actual name of the quality score column in your data.

**Main Function:**

* In the main function, make sure the filename you provide for loading the data matches the actual file you're using. For example, if your data is in "Winequality-red.csv", update the line to:

```python
def main():
  X, y = load_and_preprocess_data('Winequality-red.csv')
```

**Tips:**

* Play around with the "K" value used for cross-validation and the number of iterations ("n_iterations") for bootstrapping. These can affect how well your model performs. The best settings will depend on the size of your data and your computer's processing power.
  
### Results
![Alt Text](Wine_img.png) ![Alt Text](boston_img.png)


