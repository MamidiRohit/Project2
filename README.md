Project: k-Fold Cross-Validation and Bootstrapping Model Selection
Overview
This project implements k-fold cross-validation and bootstrapping. Both are implemented from scratch without using libraries like sklearn.model_selection.

It is based on the concepts explained in Sections 7.10–7.11 of Elements of Statistical Learning (2nd Edition), focusing on validation and model assessment techniques.

Implemented Methods

1. k-Fold Cross-Validation
Splits the dataset into k equally-sized folds.
Trains the model on k-1 folds and evaluates it on the remaining fold.
Repeats this process k times, ensuring each fold serves as the validation set exactly once.
Returns the average performance metric across all folds.

2. Bootstrapping
Generates B bootstrap samples by randomly sampling the dataset with replacement.
Trains the model on each bootstrap sample and evaluates it on the out-of-bag (OOB) samples (data points not included in the bootstrap sample).
Returns the average performance metric across all iterations.

Questions Answered
1. 
Yes, for linear regression (a simple case), the results of k-fold cross-validation and bootstrapping align well with AIC.

AIC measures model quality by penalizing model complexity, while k-fold cross-validation and bootstrapping estimate the prediction error directly.
When tested on synthetic regression data, all methods provided consistent results.

2. 
The methods may struggle in the following scenarios:

Small datasets: k-Fold Cross-Validation may produce unstable results due to limited training data in each fold.
Bootstrapping limitations: With small datasets, the out-of-bag (OOB) samples might become too small to reliably estimate error.
Imbalanced datasets: For classification problems, the class distribution may not be preserved in the splits, leading to biased results.

3. 
Stratified k-Fold Cross-Validation: Ensures class distribution in classification tasks.
Confidence intervals for bootstrapping: Provides error bounds for the performance metrics.
Custom metrics: Allow users to define their own evaluation metrics, tailored to specific tasks.
Time-series validation: For sequential data, implement time-aware validation methods like sliding windows or blocked CV.

4. 
k-Fold Cross-Validation:
k: Number of folds (default: 5).
metric: Error metric to evaluate model performance (default: Mean Squared Error).
Bootstrapping:
B: Number of bootstrap iterations (default: 10).
metric: Error metric to evaluate model performance (default: Mean Squared Error).
Connection to Elements of Statistical Learning
This implementation is inspired by Sections 7.10–7.11, which highlight:

Validation techniques like cross-validation for estimating prediction error.
Bootstrapping as a way to estimate the variability of model parameters and predictions. Both methods provide robust tools for model evaluation, particularly when the dataset is limited.
Testing
The methods were tested using:

Synthetic Data: Generated with make_regression to ensure functionality.
Linear Regression Model: Demonstrated compatibility with simple models and comparison with AIC.
Example results (Mean Squared Error):

k-Fold Cross-Validation: 0.01
Bootstrapping: 0.012
The small error values confirm that the methods perform well under ideal conditions.

Limitations
For small datasets, additional techniques like stratified sampling are recommended.
Bootstrapping may fail to generate meaningful OOB samples in extremely small datasets.
Further testing is needed on real-world datasets with complex models.
