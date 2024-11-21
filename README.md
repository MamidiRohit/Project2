## Team Members
1. Ayyappa Boovadira Kushalappa **(A20555146)**
2. Binaal Bopanna Machanda Beliappa **(A20599896)**
3. Vikas Ravikumar Karjigi **(A20581687)**
4. Yashas Basavaraju **(A20586428)**

# Model Selection

## Overview
This project provides an implementation of two fundamental model selection techniques: So, the methods are the following two: **k-fold cross-validation** and **bootstrapping**. These methods are intended to assess and compare abilities of machine learning models on the data that they have never seen before, which may provide information on how well a particular model performs when confronted with new inputs, and how accurately a model can predict outcomes. The implementation is generic and not tied to any specific model and allows users to tune it to one’s needs and budgets by adjusting its parameters.

## Features

### K-Fold Cross-Validation
- **Flexible Configuration:** Determine the number of folds, i.e. the number of sets that will be used to estimate the model in order to tune the degree of bias/variance.`
- **Shuffling Option:** It is recommended that before splitting the dataset, it should be first shuffled, so as to minimize selection bias.
- **Reproducibility:** Optional Parameter: random_state as this is used to initialized the random number generator for reproduction of the partitions of the data for testing and evaluation.

### Bootstrapping
- **Custom Sampling:** Decrease the frequency and the size of the bootstrap samples in relation to the dataset you are testing.
- **With Replacement:** Any individual datum may also be repeated within one sample, allowing for more reliable resampling.
- **Reproducibility:** It is recommended you set a means of `random_state` for deterministic and reproducible purposes.

---

## Frequently Asked Questions

### 1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
In simple cases such as linear regression, cross validation and boot strap can give out the same results as other simpler selection techniques such as AIC (Akaike Information Criterion).
- **Why They Align** :
AIC tries to compare the complexity of the model with the fitness of the model by using a penalty term that is the number of parameters in the model. AIC in simple models like the linear regression model frequently identifies models that eliminate overfitness in a model at the same time. In the same way, cross-validation estimates how the model performs on new data through an empirical evaluation of predictive accuracy of the model on a validation set which is generally more predisposed to choosing simple models when data is scanty or stochastic. Another technique that is used, often in combination with cross-validation, which involves re-sampling the dataset, and estimating the performance measure on different test sets, is also bootstrapping, which naturally prefers models of intermediate complexity and good fit.

- **Why They May Differ** :
The main weakness of AIC is that it depends on approximation of the log likelihood function of the model commonly used, which sometimes is assumed without necessarily having basis. However, cross validation and bootstrapping are more flexible methods and does not put such assumption like the above methods. Therefore in situations where these assumptions are violated (for instance when the error terms are non Gaussian) cross validation or boot strap could give results that are different from AIC.


### 2. In what cases might the methods you've written fail or give incorrect or undesirable results?
While cross-validation and bootstrapping are robust methods, they have limitations and can produce undesirable results under certain circumstances:
- **Imbalanced Datasets:**
Cross validation is potentially misleading in the situations where the number of instance in some folds is relatively small in case of the datasets with imbalanced classes. Because F1 measure averages the precision and recall rates this can make the model look better than everyone else in most of the majority class but worst in the minority class.

- **Computational Complexity:**
Despite this, both methods, in particularly in their generic form, require significant computational resources for large dataset or complex models and may be thus not suitable for real-time application or where computational power is limited.


### 3. What could you implement given more time to mitigate these cases or help users of your methods?
Several enhancements can be implemented to address the limitations and improve the robustness and usability of the methods:
- **Stratified K-Fold Cross-Validation:** It is valuable for balanced datasets but especially helpful when the imbalanced nature of the data should be preserved in each fold.

- **Early Stopping and Regularization:** Early stopping or penalization during training within cross-validation, some of the mechanisms can minimize overfitting risks especially within complex forms of the estimate model.


### 4. What parameters have you exposed to your users in order to use your model selectors?
The implementation provides flexibility by exposing the following parameters:
- **For K-Fold Cross-Validation:**
  - **k**: It shows the number of divisions that should be made to splits the dataset into folds. The default value for them is 5 or 10 that is depending on the number of records in the involved data set.
  - **alpha**: An additional parameter enables the comparison of overfitting or underfitting while evaluating the model. It modifies the proportion that is used in the penalty terms in the model if any or acts as a maker during the evaluation process. The default value is model specific but could be adjusted according to the values that are most suitable for a certain dataset and application.

- **For Bootstrapping:**
  - **n_samples**: This is the number of Bootstrap samples to be produced as indicated in the balanced Bootstrap algorithm above. Default is 1000 so that the sample size is big enough for accurate estimation.
  - **alpha**: A coefficient added to the formulas computed during the bootstrapping process in order to scale down the impact of individual samples. Resampling over specified data provides good control of overfitting or underfitting of the models as handled by this parameter. OpenClip’s default value is 10 and it can be adjusted for all models according to the dataset and the evaluation required.
    
The ability to configure these parameters, makes the methods versatile enough to handle a wide range of datasets and users needs for any other machine learning processes.

## Installation
To get started with this project, follow the steps below to install the necessary dependencies:
### 1. Clone the Repository
```bash
git clone https://github.com/vikasrkarjigi/Project2.git
cd model_selection
```
### 2. You can install the required libraries using pip:
```bash
pip install numpy matplotlib pandas
```
### 3. Run the Tests
After installation, you can run the test cases as follows:
```bash
python test_model_selector.py
```
This will run all the test cases and print the results for each dataset, including the Mean Squared Error (MSE) from cross-validation and bootstrapping, and the AIC score.

### Basic Usage
To use the Model Selector, follow the example below:
```bash
import pandas as pd
from Project2.model_selection.generic_crossvalidation import ModelSelector

def test_diabetes():
    data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
                       header=None)
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    selector = ModelSelector()

    model, alpha = "linear", 0.1
    k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)
    bootstrap_mse = selector.bootstrapping(X, y, num_samples=150, model_type=model, alpha=alpha)
    beta = selector.fit_linear_regression(X, y)
    aic_score = selector.aic(X, y, beta, model)

    results = {model: {"k_fold_mse": k_fold_mse, "bootstrap_mse": bootstrap_mse, "aic_score": aic_score}}

    print(f"\nModel: {model.capitalize()}")
    print(f"K-Fold MSE: {k_fold_mse:.4f}, Bootstrap MSE: {bootstrap_mse:.4f}, AIC: {aic_score:.4f}")

    selector.best_model(results)
    selector.visualize_model_comparison(results, "Test Diabetes")

if __name__ == '__main__':
    test_diabetes()
```

## Test Cases Outputs

Here are the test cases that demonstrate how to use the ModelSelector class on different datasets. Each test case includes cross-validation, bootstrapping, and model comparison.

1. **Wine Quality Dataset**  
   **Description**: This test evaluates the performance of different regression models (linear, ridge, and lasso) on the Wine Quality dataset, using 5-fold cross-validation and bootstrapping methods. It also calculates the AIC score for model comparison.

    ```python
   test_wine_quality()
   
**OUTPUTS**:
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/Wine_quality_test_result_1.png" alt="Wine Quality Test Result" width="500" height="200"/>
  <img src="results/Wine_quality_test_result_2.png" alt="Wine Quality Test Result" width="500" height="200"/>
</div>
<br>
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="test/Test Wine Quality.png" alt="Test Wine Quality" width="500" height="300"/>
  <img src="test/Test Wine Quality_AIC.png" alt="Test Wine Quality AIC" width="500" height="300"/>
</div>    

   
2. **Boston Housing Dataset**  
   **Description**: This test evaluates the models (linear, ridge, and lasso) on the Boston Housing dataset. It computes the k-fold cross-validation MSE, bootstrapping MSE, and AIC score for each model.

    ```python
   test_boston_housing()
   
**OUTPUTS**:
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/Boston_housing_dataset_test_1.png" alt="Boston Housing Dataset Test Result" width="500" height="200"/>
  <img src="results/Boston_housing_dataset_test_2.png" alt="Boston Housing Dataset Test Result" width="500" height="200"/>
</div>
<br>
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="test/Test Boston Housing.png" alt="Test Boston Housing" width="500" height="300"/>
  <img src="test/Test Boston Housing_AIC.png" alt="Test Boston Housing AIC" width="500" height="300"/>
</div>


3. **Diabetes Dataset**
   **Description**: This test performs regression on the Pima Indians Diabetes dataset, comparing the models using cross-validation, bootstrapping, and AIC score.

    ```python
   test_diabetes()
   
**OUTPUTS**:
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/Diabetes_dataset_test_1.png" alt="Diabetes Dataset Test Result" width="500" height="200"/>
  <img src="results/Diabetes_dataset_test_2.png" alt="Diabetes Dataset Test Result" width="500" height="200"/>
</div>
<br>   
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="test/Test Diabetes.png" alt="Test Diabetes" width="500" height="300"/>
  <img src="test/Test Diabetes_AIC.png" alt="Test Diabetes AIC" width="500" height="300"/>
</div>

4. **Concrete Strength Dataset**
   **Description**: This test assesses the concrete compressive strength dataset, evaluating linear, ridge, and lasso regression models using cross-validation, bootstrapping, and AIC score.

    ```python
   test_concrete_strength()
   
**OUTPUTS**:
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/Concrete_strength_dataset_test_1.png" alt="Concrete Strength Dataset Test Result" width="500" height="200"/>
  <img src="results/Concrete_strength_dataset_test_2.png" alt="Concrete Strength Dataset Test Result" width="500" height="200"/>
</div>
<br>   
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="test/Test Concrete Strength.png" alt="Test Concrete Strength" width="500" height="300"/>
  <img src="test/Test Concrete Strength_AIC.png" alt="Test Concrete Strength AIC" width="500" height="300"/>
</div>    

5. **Polynomial Regression**
   **Description**: This test evaluates polynomial regression models (with degree 2) on synthetic data to compare the performance of linear, ridge, and lasso regression models.
 
   ```python
   test_polynomial_regression()
   
**OUTPUTS**:
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/Polynomial_regression_test_1.png" alt="Polynomial Regression Test Result" width="500" height="200"/>
  <img src="results/Polynomial_regression_test_2.png" alt="Polynomial Regression Test Result" width="500" height="200"/>
</div>
<br>   
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="test/Test Polynomial Regression.png" alt="Test Polynomial Regression" width="500" height="300"/>
  <img src="test/Test Polynomial Regression_AIC.png" alt="Test Polynomial Regression AIC" width="500" height="300"/>
</div> 


## Additional Features
Several additional features have been implemented to enhance the model selection process:
- **Added Lasso and Ridge with Normal Linear Regression**: Lasso and Ridge regression models have been added alongside the basic linear regression model to provide regularization techniques that help in controlling overfitting.
  
- **Comparison of AIC Values**: The AIC values for all models (linear regression, ridge, and lasso) are calculated and compared to determine the best model for the specific dataset. The model with the lowest AIC value is selected as the best performing model.
- **Visualization of Model Performance**: A bar graph is plotted comparing the k-fold cross-validation MSE, bootstrap MSE, and AIC for all three models (linear regression, ridge, and lasso). This visualization helps users compare the performance of different models on the same dataset.
- **Test Cases with Special Features**: Five test cases are implemented, each with a different special feature to assess model performance under different conditions. These include datasets with varying complexity and characteristics such as non-linearity, outliers, and imbalanced features.

### Implemented Two New Classes:
- **BestModel**: This class is designed to return the best model for a given dataset based on the lowest AIC score.
  
- **VisualizeModelComparison**: This class visualizes the comparison between models by plotting a bar graph of k-fold cross-validation MSE, bootstrap MSE, and AIC for all models, offering better insight into model performance.


## Conclusion
Cross validation and bootstrapping are imperative procedure integrated into this work, to provide consistent and reliable estimation, and accurate decision-making towards the selection of the most suitable model. They are a more accurate way of generating preliminary estimates of the accuracy of a model across different data sets, and, in general, of how well a model will generalize to new data. The use of K-fold cross-validation provides insights for all data points both as a training set and as a test set, giving a holistic view of model potentialities; bootstrapping, on the other hand, resorts to resampling in order to measure stability and variability of chosen indexes. These methods are helpful in checking the accuracy of the model and minimize the biases, so that the variance is utilized to determine the way, better models perform with new numbers and inputs. It also employs ordinary, linear and ridge, and L1 for comparing the efficiency with its AIC scores across the datasets.

However, as was illustrated in the project, these methods have drawbacks, including high computational complexity with large datasets, and unsuitability for small or imbalanced datasets. Some of these problems can be solved including; changing the number of folds or the bootstrap samples, use of stratified data, and making corrections on the values of k and alpha. This is precisely why the application of these techniques has been brought into this project to demonstrate where and when such approaches may be useful for developing practical, reliable and accurate and generalizable machine learning models that data scientists operationalize for decision making purposes.
