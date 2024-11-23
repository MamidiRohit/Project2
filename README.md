Team: Data Dynamos
Project Members:
Rohit Kumar Mamidi(A20541036)
Ganesh Maddula(A20541032)

Contributions:
Rohit Kumar Mamidi
Developed and implemented K-Fold Cross-Validation and Bootstrapping model selection methods.
Conducted performance evaluation and designed modular scripts for data generation and training.
Debugged the testing pipelines for error-free execution.

Ganesh Maddula
Designed and implemented Linear Regression and Dataset generation modules.
Created and prepared final evaluation metrics for K-Fold and Bootstrapping methods.
Led the documentation and testing framework integration.
Both members contributed equally to brainstorming and refining the model selection process.

Requirements:
Python Version: 3.8+
Required Libraries:
numpy
pandas
pytest


Project Overview
This project, Efficient Model Selection Techniques, focuses on implementing and comparing two robust statistical methods for model evaluation: K-Fold Cross-Validation and Bootstrapping. These methods ensure the generalizability and reliability of predictive models, particularly in regression tasks, and allow for meaningful insights into their performance across diverse datasets.

By addressing practical challenges in model selection, such as overfitting, variance reduction, and computational efficiency, we aim to provide a flexible and user-friendly framework to evaluate and compare models effectively.

Key Features
Model Selection Methods:

K-Fold Cross-Validation: Splits the dataset into k subsets (folds) and evaluates the model k times, ensuring fair performance assessment.
Bootstrapping: Generates multiple resampled datasets to estimate performance and variability across iterations.

Versatile Data Handling:

Supports generated datasets with configurable parameters (size, noise, and seed).
Accepts predefined datasets like those generated from professors’ algorithms.
Allows evaluation on custom CSV datasets (e.g., test_data.csv).

Performance Metrics:

Mean Squared Error (MSE): Quantifies prediction error.
Mean Absolute Error (MAE): The average magnitude of errors in predictions.
Root Mean Squared Error (RMSE): The square root of the average squared difference between predictions and actual values.
R2 Score: Evaluates variance explained by the model.
Akaike Information Criterion (AIC): Balances model fit against complexity.

Extensibility:

Modular design enables integration with other machine-learning models beyond linear regression.
Parameterized for custom evaluation scenarios.

Setup Instructions:
Clone the repository and install the necessary libraries:
https://github.com/MamidiRohit/Project2.git
pip install -r requirements.txt

Running the test file
python test.py
python run_model.py

Synthetic Data Arguments:
Number of Samples (N): Specifies how many data points (samples) will be generated for testing and training the models.

Regression Coefficients (m): These are the coefficients used in the linear regression equation to define the strength of the relationship between each feature and the target variable. For instance, a simple linear regression model could have one coefficient per feature, and the target variable 𝑦

Intercept (b): The intercept is the baseline value of the target variable when all feature values are zero. This is part of the linear regression equation.

Noise (noise_level): Adds randomness or variability to the data to simulate real-world imperfections. This helps prevent overfitting to perfectly linear data.

Feature Range (rnge): Specifies the range within which the feature values should fall. For example, the features might be generated in the range between 0 and 10.

Random Seed (random_seed): Ensures that the data generation process is reproducible across different runs by fixing the random number generation process.

Generated Data Arguments:
--rows:
Description: Specifies the number of rows (samples) in the generated dataset.
Example: --rows 2000 means the dataset will contain 2000 samples.

--cols:
Description: Specifies the number of columns (features) in the generated dataset.
Example: --cols 20 means the dataset will have 20 features for each sample.

--noise_level:
Description: Controls the noise level added to the data to simulate randomness and real-world imperfections. This is typically added to the target variable to prevent perfect linear relationships.
Example: --noise_level 0.5 means noise will be added with a scale of 0.5.

--random_seed:
Description: Sets the random seed to ensure reproducibility of the generated data. This allows the dataset to be regenerated with the same random values each time.
Example: --random_seed 35 ensures that the random number generator used to generate data is consistent across runs.

Questions:
1.Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
In simple cases like linear regression, where the model is not overly complex, the results from K-Fold Cross-Validation and Bootstrapping typically align with the model selection suggested by AIC. Both K-Fold and Bootstrapping are designed to evaluate how well a model generalizes to new, unseen data by repeatedly splitting the data into training and test sets. They provide performance metrics like Mean Squared Error (MSE) and R², which are useful for assessing model accuracy.

AIC, on the other hand, focuses on balancing model fit and complexity. It penalizes models with more parameters to avoid overfitting, preferring simpler models that still fit the data well. In cases like linear regression, where the model complexity is low and the number of parameters is small, K-Fold and Bootstrapping methods will likely produce similar results to AIC, as they all support the selection of models that both generalize well and fit the data adequately.


2.In what cases might the methods you've written fail or give incorrect or undesirable results?
The methods used in our project may fail or give undesirable results in several scenarios:

1.Multicollinearity in the data can lead to instability in linear regression models, inflating coefficients and distorting predictions.
2.Non-linear relationships between features and the target can make linear regression inadequate, causing poor model performance.
3.For small datasets, bootstrapping may fail due to insufficient out-of-bag samples, leading to unreliable estimates.
4.Lack of diversity in bootstrapped samples can result in inaccurate model validation if the data does not represent the overall population.
5.Imbalanced datasets can cause biased performance metrics in K-fold cross-validation, as some classes may dominate certain folds.
6.With limited data, K-fold results can vary significantly, making it difficult to assess the model’s true performance consistently.


3.What could you implement given more time to mitigate these cases or help users of your methods?
1.Handling Multicollinearity: Implement feature selection techniques, such as Principal Component Analysis (PCA) or Lasso regularization, to reduce multicollinearity and stabilize coefficient estimates in linear regression models.

2.Non-Linear Relationships: Introduce more advanced models like decision trees, random forests, or support vector machines (SVMs) to capture non-linear patterns, ensuring better performance on non-linear data.

3.Bootstrap with Small Datasets: Enhance bootstrapping by using Stratified Bootstrapping or increasing the number of bootstrap iterations to improve the accuracy and reliability of out-of-bag estimates for small datasets.

4.Data Augmentation: For sparse datasets, implement data augmentation techniques to artificially expand the dataset, making the bootstrapping process more effective by introducing more diversity.

5.Model Performance Monitoring: Add cross-validation performance tracking to monitor the stability of the evaluation metrics across multiple runs, helping identify variability in small datasets and offering more robust performance estimates.

6.Outlier Detection and Handling: Implement methods to detect and handle outliers or anomalies in the data that could skew the model's performance and the evaluation metrics.


4.What parameters have you exposed to your users in order to use your model selectors?
For Linear Data Generator (ProfessorData class):

Synthetic Data Arguments
rnge: Range of values for features.
-random_seed: Seed for reproducibility.
-N: No of samples.
-m: Regression coefficients.
-b: Offset.
-scale: Scale of noise.

Data Generator Arguments:
cols: Number of features in the generated data from data generator.
--noise: Noise level in the generated data.
--seed: Random seed for reproducibility.
--rows: Number of samples in the generated data.

Results:
Kfold:
Mean Squared Error (MSE): 270171044.1141
R² (Coefficient of Determination):  0.8401
Mean AIC: 1948.4523
Mean Absolute Error (MAE): 13006.6129
Root Mean Squared Error (RMSE): 16395.1934

BootStrapping:
Mean Absolute Error (MAE): 13159.8830
Mean Squared Error (MSE): 279085329.9941
Root Mean Squared Error (RMSE): 16693.5849
R² (Coefficient of Determination): 0.8372
Akaike Information Criterion (AIC): 3698.0397
