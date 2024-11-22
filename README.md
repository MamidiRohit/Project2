# Project 2 : Model Selection 

# Team: Data Mavericks

## Implement generic k-fold cross-validation and bootstrapping model selection methods.

### Overview:

This project implements Ridge Regression for predicting airplane prices based on critical features such as engine rate of climb, takeoff distance, and range. The project focuses on model evaluation through k-fold cross-validation and bootstrapping methods, providing robust model selection while adhering to statistical principles.

Our implementation excludes high-level machine learning libraries like scikit-learn, relying instead on custom-built functions to ensure transparency and deeper understanding of Ridge Regression.


### How to Run the Code :- 

1. Clone the repository or download the notebook and data files.

2. Ensure the dataset is saved in the same directory as the notebook. Hardcode the dataset path in the notebook if required (e.g., df = pd.read_csv('plane Price.csv')).

3. Install the required libraries:

```python
pip install numpy
pip install pandas
pip install statsmodels
pip install seaborn
pip install matplotlib
```

4. Execute the cells step by step to preprocess data, train models, and evaluate performance.

5. The last code block allows you to input numerical values for 'engine rate of climb', 'takeoff over 50ft', and 'range' to get a predicted airplane price.


## Answers to README Questions :

### 1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)? 

### Ans :

Yes, the cross-validation and bootstrapping results align closely with the AIC (Akaike Information Criterion) approach in simple cases like linear regression. 
The following observations were noted :

• Both methods confirmed the same regularization strength (alpha) as optimal during Ridge Regression hyperparameter tuning.

• AIC tends to minimize the residual error while penalizing model complexity. Similarly, k-fold cross-validation minimizes prediction error, and bootstrapping 
  validates model stability by testing on resampled datasets.
  
• Thus, in simple cases, the model selectors provide consistent results and help confirm that the regularization and model complexity are well-balanced.


### 2. In what cases might the methods you've written fail or give incorrect or undesirable results?

### Ans :

The methods may fail or provide undesirable results in the following scenarios:

• Non-linear Relationships: Ridge Regression assumes linear relationships between predictors and the target. It might not function effectively if the underlying.
  
• Outliers in the Dataset: Although Ridge reduces the impact of multicollinearity, it does not address outliers, which may skew predictions.

• Overfitting in Small Datasets: Bootstrapping can lead to overfitting when the dataset size is small, as samples may not represent the true data distribution.

• Multicollinearity in k-fold Splits: If the feature selection is not consistent across folds, k-fold cross-validation may produce unstable results.



### 3. What could you implement given more time to mitigate these cases or help users of your methods?

### Ans :

With additional time, the following enhancements could be implemented:

• Polynomial Feature Transformation: To account for non-linear relationships between features and the target variable.

• Outlier Detection: Add preprocessing steps to detect and handle outliers, such as robust scaling or removing extreme values.

• Stratified Cross-Validation: Ensure more representative splits in k-fold cross-validation, particularly for datasets with imbalanced distributions.

• Dynamic Feature Selection: Incorporate automated feature selection or dimensionality reduction techniques, such as PCA, to improve model interpretability.



### 4. What parameters have you exposed to your users in order to use your model selectors?

### Ans :- 
The implementation exposes the following parameters for users to fine-tune the model:

1. Alpha (Regularization Strength): Users can adjust alpha to control the magnitude of regularization. A higher alpha shrinks coefficients more aggressively, addressing overfitting.

2. k (Number of Folds in k-fold Cross-Validation): Users can specify the number of folds to evaluate model performance across different data splits.

3. Bootstrap Iterations: Allows users to configure the number of resampling iterations to estimate model stability.

4. Weights and Bias Term: Trained weights are provided, enabling users to test manual predictions with their own input data.




### Libraries Used :

• NumPy: For numerical computations, including matrix operations and random sampling for bootstrapping.

• Pandas: For data manipulation and preprocessing.

• Matplotlib and Seaborn: For data visualization, including plotting the correlation matrix, residual analysis, and performance evaluations.

• Statsmodels: To compute Variance Inflation Factor (VIF) for multicollinearity analysis.



### Key Components of the Code:

1. Correlation Matrix:

     • Purpose:  Identifies relationships between variables and highlights features strongly correlated with Price.
   
     • Why:  Aids feature selection and avoids multicollinearity, improving model efficiency.

2. VIF Analysis:

     • Purpose:  Measures multicollinearity and removes variables with high VIF.
     
     • Why:  Ensures stable and interpretable model coefficients.

3. Ridge Regression:

     • Purpose:  Regularizes the model to handle multicollinearity and prevent overfitting.
     
     • Why:  Enhances generalizability by balancing bias and variance.

4. Hyperparameter Tuning:

     • Purpose:  Fine-tunes alpha using cross-validation for optimal regularization.
     
     • Why:  Achieves the best trade-off between bias and variance.

5. Bootstrapping:

     • Purpose: Validates model stability by evaluating R² across multiple resampled datasets.
     
     • Why: Ensures consistent performance under different conditions.

6. Adjusted R²:

     • Purpose:  Evaluates model fit while penalizing unnecessary complexity.
     
     • Why:  Prevents overfitting by adding irrelevant predictors.

7. Visualization:

     • Purpose:  Displays predicted vs. actual prices and residual analysis to evaluate model accuracy.
     
     • Why:  Demonstrates the goodness of fit and identifies potential deviations





###  Airplane Price Predictor:

1. Purpose:

    • An interactive module that predicts airplane prices based on user input.
   
    • Shows how the model is used in the real world.

3. How It Works:

    • Inputs: Accepts user specifications for features like rate of climb, takeoff distance, and range.
   
    • Scaling: Standardizes inputs using precomputed means and standard deviations.
   
    • Prediction: Applies the trained Ridge Regression model to compute the price.


5. Why Include It:

   • Practical Utility: Showcases how the model can be used for decision-making in aviation pricing.
   
   • Stakeholder Engagement: Provides an interactive way to demonstrate the model’s relevance. 



### Code Usage: Example of Using Code :

The project involves multiple steps for data preprocessing, model training, and evaluation. Below is an example of how to use the code:

#### Run the Ridge Regression Model:

• Load the dataset and preprocess it (e.g., handle missing values, scale features, compute the correlation matrix).

• Train the Ridge Regression model with different alpha values to identify the optimal regularization strength.

#### Hyperparameter Tuning:

• Use k-fold cross-validation to evaluate model performance across splits.

• Perform bootstrapping to validate model stability under resampling conditions.

#### Manual Prediction:

In the final block titled "Airplane Price Predictor," you can input airplane specifications to get a price prediction.

Example:

```python
# Example Inputs :
Enter engine rate of climb (ft/min): 1800
Enter takeoff distance over 50ft (ft): 4800
Enter range (nautical miles): 2000

# Output :
Predicted Price: $2,750,000.00
```


### Visualization of Results :

The project uses data visualization extensively to interpret results and validate model performance. Below are the key visual outputs included in the project:

1. Correlation Matrix:

   • A heatmap visualizing the relationships between features.

   • Example: Features like "Takeoff Distance" and "Engine Rate of Climb" show strong correlations with the price, aiding in feature    selection.

2. Residual Analysis:

   • Residual plots evaluate the accuracy of predictions.

   • Example: Residuals scatter symmetrically around zero, indicating a well-fitted model.

3. Alpha Tuning (Ridge Regression):

   • A graph shows R² values for different regularization strengths (alpha) during hyperparameter tuning.

   • Example: The curve helps identify the alpha value that provides the best trade-off between bias and variance.

4. Predicted vs. Actual Prices:

   • A scatter plot comparing model predictions with actual prices.

   • Example: Points align closely to the diagonal "Perfect Fit" line, demonstrating good model predictions.


### Contribution

Kaustubh Dangche - A20550806
 Data Cleaning and Preprocessing with VIF Analysis
The data cleaning process removes inconsistencies and handles missing values to ensure quality. The Variance Inflation Factor (VIF) is calculated to identify and remove highly collinear features, improving model stability and interpretability. This step ensures the selected features are relevant for predicting airplane prices.
________________________________________
Hyunsung Ha - A20557555
2. Feature Scaling, Train-Test Split, and Regression Models
Feature scaling standardizes input variables, ensuring uniform contribution to the model. The dataset is split into training and testing sets to evaluate performance on unseen data. Ridge and Lasso regression models are compared, with Ridge emphasizing multicollinearity handling and Lasso favoring sparse feature selection.
________________________________________
Anu Singh - A20568373
3. K-Fold Cross-Validation and Bootstrapping
K-fold cross-validation evaluates model performance across multiple data splits, ensuring robustness and reliability. Bootstrapping validates the model's stability by testing it on resampled datasets, providing additional confidence in its generalizability.
________________________________________
Nam Gyu Lee - A20487452
4. Visualization and the Plane Price Predictor
Data visualization, including correlation matrices, residual plots, and predicted vs. actual price comparisons, highlights relationships and model accuracy. The interactive airplane price predictor allows users to input specifications and get real-time predictions, demonstrating the model's practical application in aviation pricing.




