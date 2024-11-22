## Amruta Sanjay Pawar- A20570864
## Raghav Shah- A20570886
## Shreedhruthi Boinpally- A20572883
# Linear Regression with Evaluation Metrics

This project demonstrates the implementation of a custom linear regression model in Python and evaluates its performance using K-Fold Cross-Validation, Bootstrapping, and Akaike Information Criterion (AIC). The project uses a dataset named `heart.csv` for testing and validation.

## Objective
To test and evaluate the performance of a custom linear regression model on the any dataset using the following metrics:
1. K-Fold Cross-Validation Mean Squared Error (MSE)
2. Bootstrapping Mean Squared Error (MSE)
3. Akaike Information Criterion (AIC)

### Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?  
Yes, in our testing with the `heart.csv` dataset using linear regression, the model selectors produced consistent results:
- **K-Fold Cross-Validation MSE**: 0.1241  
- **Bootstrapping MSE**: 0.1254  
- **AIC**: -2137.2136  

These results indicate that cross-validation, bootstrapping, and AIC agree when evaluating the model’s fit to the data for simple cases.

---

### In what cases might the methods you've written fail or give incorrect or undesirable results?  
1. **Small datasets**: Bootstrapping may produce biased results due to repeated sampling from limited data.  
2. **Imbalanced data**: K-Fold Cross-Validation may fail if folds are not stratified.  
3. **High-dimensional data**: AIC may over-penalize complex models, leading to biased selection.  
4. **Outliers**: All methods may give undesirable results if outliers dominate the dataset.  

---

### What could you implement given more time to mitigate these cases or help users of your methods?  
- Implement **Stratified K-Folds** for handling imbalanced datasets.  
- Incorporate **robust regression techniques** to handle outliers effectively.  
- Use **Bayesian Information Criterion (BIC)** alongside AIC for high-dimensional datasets.  
- Add automated **dataset analysis and warnings** for users regarding dataset size, balance, or presence of outliers.  

---

### What parameters have you exposed to your users in order to use your model selectors?  
- **K-Fold Cross-Validation**:  
  - `k`: Number of folds.  
  - `metric`: Metric to evaluate (e.g., MSE or R²).  
  - `random_seed`: Seed for reproducibility.  

- **Bootstrapping**:  
  - `num_samples`: Number of bootstrap samples.  
  - `metric`: Metric to evaluate (e.g., MSE or R²).  
  - `random_seed`: Seed for reproducibility.  

- **AIC**:  
  - Requires no additional parameters and uses the full dataset.

These parameters allow users to adapt the methods to their specific datasets and modeling requirements.


## How to Run
1. Clone this repository
2. Install required Python libraries (`matplotlib` for plotting) from cmd.
3. Replace with your CSV file path in model.py
4. Replace with your target column name in model.py
5. Then run model.py file

