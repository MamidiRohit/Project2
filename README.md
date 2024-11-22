# Generic k-fold cross-validation and bootstrapping model selection methods Implementation

## Team: Model Masters

## Project Members
The following members worked on this project:

- **Atharv Patil (A20580674)**
- **Emile Mondon (A20600364)**
- **Merlin Santano Simoes (A20531255)**
- **Tejaswini Viswanath (A20536544)**

## Contributions

- Model Implementation and Algorithm Development [Atharv, Tejaswini]
- Model Selection and Pytest Implementation [Merlin, Emile]
- Visualization and Documentation [Everyone]
- **Everyone contributed to the project equally**

### Requirements

- Python 3.7+
- Required Libraries: `numpy`, `pandas`, `pytest`

### File Structure

- `Models/Bootstrapping.py`: Contains the `Bootstrapping Model` class to perform bootsrapping.
- `Models/Kfold.py` : Performs k-fold cross-validation.
- `Models/LinearRegression.py` : Contains the `Linear Regression Model`.
- `Data/Data_Gen.py`: Contains `DataGenerator` and `ProfessorData` classes to generate datasets.
- `Notebooks/visualization.ipynb`: Contains the visualizations for the models.
- `Test/Test.py`: Main script to train the bootstrapping and kfold models and evaluate performance metrics on csv data, professor's code generated data and our own generated data.
- `Test/Py_Test.py`: Script to run tests for model and data functions using PyTest. It includes 5 test cases to test all the models.

---

### Setup

Clone the repository and ensure the required libraries are installed:
```bash
git clone https://github.com/your-repo-name/k-fold-and-bootstrapping
pip install -r requirements.txt
```

### How to Run Test.py

`Test.py` accepts several arguments that allow you to configure data generation, K-fold parameters, and CSV file inputs. Below are instructions and examples for using each argument. 
- Note that use only one type of arguments at a time 

## Argument Options

### Generated Data Arguments:

- `--rows`: Number of rows/samples in the generated data.
- `--cols`: Number of columns/features in the generated data.
- `--noise`: Noise level in the generated data.
- `--seed`: Random seed for reproducibility.

### Example Command:

```bash
python Test.py --rows 100 --cols 10 --noise 0.1 --seed 42
```

### Professor Data Generation Arguments:

- `-N`: Number of samples.
- `-m`: Regression coefficients.
- `-b`: Offset (intercept).
- `-scale`: Scale of noise.
- `-rnge`: Range of values for features.
- `-random_seed`: Seed for reproducibility.

### Example Command:

```bash
python Test.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42
```

### CSV File Input Arguments:

- `--csv_file_path`: Path to the CSV file containing your dataset.
- `--target_column`: Name of the target column in the CSV file.

### Example Command:

```bash
python Test.py --csv_file_path "Data/35.waves.csv" --target_column "WVHT" -k 10 -n_iter 20
```
### K-Fold Model Arguments:

- `-k`: Number of Folds

## Bootstrapping Model Arguments:

- `-n_iter`: Number of Iterations

### Example Command:

```bash
python Test.py --rows 1000 --cols 10 --noise 0.2 -k 5 -n_iter 20
```


### Example Commands

### Generate Data and Train Model:

```bash
python Test.py --rows 1000 --cols 10 --noise 0.2 -k 5 -n_iter 20
```
### Train Model on Data Generated from Professor Code

```bash
python Test.py  -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42 -k 5 -n_iter 20
```

### Train Model on CSV Data:
```bash
python Test.py --csv_file_path "data/small_test.csv" --target_column "y" -k 5 -n_iter 20
```
### Running Tests


To verify the functionality, you can run the test script `Py_Test.py`, which includes unit tests for functions in the model pipeline. Make sure you are in the Test Directory before running the below command.


```bash
pytest Py_Test.py
```

## Output
The script outputs the following evaluation metrics:

- **Mean Squared Error (MSE)**: Quantifies the prediction error by averaging the squares of the differences between the actual and predicted values. A lower MSE indicates better model performance.

- **R2 Score**: Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. An R2 score closer to 1 indicates a better model fit.

- **Akaike Information Criterion (AIC) Score**: AIC is a model selection metric that evaluates the trade-off between model fit and complexity.


## Q1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
- K-fold cross-validation and bootstrapping methods used to evaluate model performance using MSE and R-Squared metrics. These metrics usually correspond to simpler model choices. such as AIC when the dataset and linear regression are appropriate (e.g. Gaussian noise There are not many straight lines. linear relationship)

- However, AIC directly penalizes model complexity. This is clearly not included in MSE and R-squared. In cases with a large number of features, cross-validation and bootstrapping can provide better results for overfitting models compared to AIC.


## Q2. In what cases might the methods you've written fail or give incorrect or undesirable results?
**Data problems:** 
- **Severe multicollinearity:** Linear Regression Model Uses normal equations which are sensitive to multi-collinearity. This leads to numerical instability and inflated weights..

- **Non-linear relationships:** The linear regression assumption may not hold. This results in poor performance regardless of the evaluation method.

**Bootstrap specific problems:**
 - **Sparse dataset:** Bootstrap validation relies on out-of-bag samples. For small data sets, OOB samples may be too few for reliable verification.

- **Lack of diversity of information:** If the distribution of information is distorted Bootstrapped sampling may not capture the variance required for effective estimation.

 **Cross Validation Specific Issues:**
- **Unbalanced dataset:** K-fold CV may not produce representative folds. This leads to biased estimates.
- **Small data sets:** Limited data can cause k-folded results to be highly variable due to fold assignment.


## Q3. What could you implement given more time to mitigate these cases or help users of your methods?
**Model durability**: 
- Use Ridge or Lasso regression to deal with multicollinearity improved
- Provide polynomial regression or feature engineering to capture non-linear relationships.

**Improving data partition**: 
- Layered k-fold CV to handle unbalanced datasets
- Weighted bootstrapping to ensure diverse sampling from underrepresented classes or regions.

**Numerical stability**: 
- Add normalization or pseudo-inverse calculation techniques to identify singular matrices in general equations.

**Additional indicators**: 
- Includes metrics such as AIC, BIC or adjusted R-Squared for better model comparison.

## Q4 What parameters have you exposed to your users in order to use your model selectors?
**For k-Fold Cross-Validation**
- `k:` Number of folds for k-fold CV, allowing users to control the trade-off between bias and variance.

**For Bootstrapping**
- `n_iterations`:Number of bootstrap iterations to control the statistical reliability of the results.

**For Linear Data Generator (ProfessorData class)**:
- `-N`: Number of samples.
- `-m`: Regression coefficients.
- `-b`: Offset (intercept).
- `-scale`: Scale of noise.
- `-rnge`: Range of values for features.
- `-random_seed`: Seed for reproducibility.

### Generated Data Arguments:

- `--rows`: Number of rows/samples in the generated data.
- `--cols`: Number of columns/features in the generated data.
- `--noise`: Noise level in the generated data.
- `--seed`: Random seed for reproducibility.

## Results

### Our Generared Data
Here are the results from the **K-Fold model** after training on Generated Data:

- **Mean Squared Error on Test Set**: 
    ```
    0.1534
    ```
- **R2 Score on Test Set**: 
    ```
    0.9796
    ```
- **AIC Score**: 
    ```
    -353.0277
    ```
Here are the results from the **Bootstrapping model** after training on Generated Data:

- **Mean Squared Error on Test Set**: 
    ```
    0.1534
    ```
- **R2 Score on Test Set**: 
    ```
    0.9796
    ```
- **AIC Score**: 
    ```
    -667.8130
    ```

### Professor Generated Data
Here are the results from the **K-Fold model** after training on Professor Generated Data:

- **Mean Squared Error on Test Set**: 
    ```
    0.1994
    ```
- **R2 Score on Test Set**: 
    ```
    0.9382
    ```
- **AIC Score**: 
    ```
    -10.8083
    ```

Here are the results from the **Bootstrapping model** after training on Professor Generated Data:

- **Mean Squared Error on Test Set**: 
    ```
    0.2285
    ```
- **R2 Score on Test Set**: 
    ```
   0.9429
    ```
- **AIC Score**: 
    ```
    -31.7819
    ```

## CSV Data

Here are the results from the **K-Fold model** after training on CSV data:

- **Mean Squared Error on Test Set**: 
    ```
    0.0137
    ```
- **R2 Score on Test Set**: 
    ```
    0.9013
    ```
- **AIC Score**: 
    ```
    -3311.0304
    ```

Here are the results from the **Bootstrapping model** after training on CSV data:

- **Mean Squared Error on Test Set**: 
    ```
   0.0139
    ```
- **R2 Score on Test Set**: 
    ```
   0.8998
    ```
- **AIC Score**: 
    ```
    -6108.5117
    ```

## PyTest Structure

We have included below 5 test cases to evaluate the functionality of the models:
- test_linear_regression
- test_k_fold
- test_bootstrapping
- test_data_generation
- test_csv_input

### Visualization of Results

- **A line chart visualizing the MSE and R-Squared scores for each **fold** to assess variability and performance across the folds**:

![K-Fold Cross-Validation Results](<Notebooks/Images/K-Fold Cross-Validation Results.png>)

- **A line chart visualizing the MSE and R-Squared scores to observe how the scores vary across bootstrap samples**:

![Bootstrapping Results](<Notebooks/Images/Bootstrapping Results.png>)

- **The following plot is the values of **MSE and RSquared Error** for the above two methods**:

![Comparison of K-Fold and Bootstrapping](<Notebooks/Images/Average Scores Comparison.png>)





