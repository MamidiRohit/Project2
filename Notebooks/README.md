# Generic k-fold cross-validation and bootstrapping model selection methods Implementation

## Project Members
The following members worked on this project:

- **Atharv Patil (A20580674)**
- **Emile Mondon (A20600364)**
- **Merlin Santano Simoes (A20531255)**
- **Tejaswini Viswanath (A20536544)**

### Requirements

- Python 3.7+
- Required Libraries: `numpy`, `pandas`, `pytest`

### File Structure

- `Models/Bootstrapping.py`: Contains the `Bootstrapping Model` class to perform bootsrapping.
- `Models/Kfold.py` : Performs k-fold cross-validation.
- `Models/LinearRegression.py` : Contains the `Linear Regression Model`.
- `Data/Data_Gen.py`: Contains `DataGenerator` and `ProfessorData` classes to generate datasets.
- `Visualizations.py`: Contains the visualizations for the models.
- `Test/test.py`: Main script to train the bootstrapping and kfold models and evaluate performance metrics.
- `PyTest.py`: Script to run tests for model and data functions using PyTest.

---

### Setup

Clone the repository and ensure the required libraries are installed:
```bash
git clone https://github.com/your-repo-name/k-fold-and-bootstrapping
pip install -r requirements.txt
```
Also, make sure to change the path in `Test.py` and `Py_Test.py`  as needed:

```bash
import sys
sys.path.insert(0, '-- path of Root Directory -- ')
```
### How to Run Model_test.py

`Test_Model.py` accepts several arguments that allow you to configure data generation, Elastic Net parameters, and CSV file inputs. Below are instructions and examples for using each argument. 
- Note that use only one type of arguments at a time 

## Argument Options

### Generated Data Arguments:

- `--rows`: Number of rows/samples in the generated data.
- `--cols`: Number of columns/features in the generated data.
- `--noise`: Noise level in the generated data.
- `--seed`: Random seed for reproducibility.

### Example Command:

```bash
python Test_Model.py --rows 100 --cols 10 --noise 0.1 --seed 42
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
python Test_Model.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42
```

### CSV File Input Arguments:

- `--csv_file_path`: Path to the CSV file containing your dataset.
- `--target_column`: Name of the target column in the CSV file.

### Example Command:

```bash
python Test_Model.py --csv_file_path "data/small_test.csv" --target_column "y"
```
### Elastic Net Model Arguments:

- `--alpha`: Regularization strength.
- `--penalty_ratio`: Ratio between L1 and L2 penalties.
- `--learning_rate`: Learning rate.
- `--iterations`: Number of iterations.

### Example Command:

```bash
python Test_Model.py --alpha 0.01 --penalty_ratio 0.1 --learning_rate 0.001 --iterations 10000
```

### Test Set:
- `--test_size`: Fraction of data to be used for testing.

### Example Commands

### Generate Data and Train Model:

For Our Generated Data
python Test.py --rows 1000 --cols 10 --noise 0.2 -k 5 -n_iter 20

for Professor Code 
Python Test.py  -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42 -k 5 -n_iter 20

For a CSV File 
python Test.py --csv_file_path "data/small_test.csv" --target_column "y" -k 5 -n_iter 20

```bash
python Test_Model.py --rows 100 --cols 5 --noise 0.2 --alpha 0.01 --penalty_ratio 0.5 --learning_rate 0.001 --iterations 5000 --test_size 0.2
```
### Train Model on Data Generated from Professor Code

```bash
python Test_Model.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0.0 10.0 -random_seed 42 --alpha 0.01 --penalty_ratio 0.5 --learning_rate 0.001 --iterations 5000 --test_size 0.2
```

### Train Model on CSV Data:
```bash
python Test_Model.py --csv_file_path "data/small_test.csv" --target_column "y"
```
### Running Tests

To verify the functionality, you can run the test script `PyTest.py`, which includes unit tests for functions in the model pipeline. Make sure you are in the Test Directory before running the below command.

```bash
pytest PyTest.py
```

## Output
The script outputs the following evaluation metrics:

- **Mean Squared Error (MSE)**: Quantifies the prediction error by averaging the squares of the differences between the actual and predicted values. A lower MSE indicates better model performance.

- **Mean Absolute Error (MAE)**: Represents the average absolute error between the actual and predicted values. Like MSE, a lower MAE signifies a better fit of the model to the data.

- **R2 Score**: Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. An R2 score closer to 1 indicates a better model fit.


## Q1a. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
Yes, in simple cases like linear regression where assumptions such as linearity, normality, and homoscedasticity are met, cross-validation and bootstrapping tend to align with AIC. Both methods evaluate models differently:
- AIC evaluates model fit and penalizes complexity, focusing on likelihood-based metrics.
- Cross-validation and bootstrapping focus on out-of-sample predictive performance.

Discrepancies may arise when:
- The dataset has noise or outliers, affecting prediction-focused methods (cross-validation, bootstrapping) more.
- Regularization changes model complexity, and this effect may not be captured by AIC as well.


### Q1b. In what cases might the methods you've written fail or give incorrect or undesirable results?
**K-Fold Cross-Validation**
- **Class Imbalance**: If the target variable is imbalanced, certain folds may not have adequate representation of all classes.
- **Small Dataset Size**: High variability across folds may lead to lower reliability if there are few overall data points per fold.

**Bootstrapping**
- **Over-Representation of Data Points**: Resampling with replacement often results in very duplicated samples, potentially biasing the results.
- **Small Dataset**: Out-of-bag samples may be sparse representatives of the full dataset, reducing the robustness of the evaluation.

**Both**
- The performance metric might be misleading if the metric used in the evaluation does not suit the problem (e.g., using accuracy when classes are imbalanced).
- It is hard for the result to generalize well if the data has some unseen pattern.


## Q3. What could you implement given more time to mitigate these cases or help users of your methods?
**For K-Fold Cross-Validation**
- Introduce **stratified splitting** so that it maintains class proportions across folds.
- Include **group splitting** to handle grouped data .

**For Bootstrapping**
- Implement **balanced bootstrapping** to enforce a variety within the sampled subsets.
- Allow the users to tweak the number of OOB samples for a more robust evaluation.

**General**
- Add options for **multiple performance metrics** such as RMSE, F1-score etc.
- Visualizations: bias-variance tradeoffs and learning curves
- Parallelised computations for larger datasets


## Q4 What parameters have you exposed to your users in order to use your model selectors.
**For k-Fold Cross-Validation**
- `n_splits`: Number of folds.
- `shuffle`: Whether to shuffle data before splitting.
- `random_state`: Seed for reproducibility.
- `scoring_metric`: Metric used to evaluate models (e.g., RMSE, accuracy).

**For Bootstrapping**
- `n_iterations`: Number of bootstrap iterations.
- `sample_size`: Size of each bootstrap sample - fraction or absolute.
- `random_state`: For reproducibility.
- `scoring_metric`: The metric used for scoring the models; examples are RMSE and accuracy.

## Results
Here are the results from the model after training on Generated Data:

- **Mean Squared Error on Test Set**: 
    ```
    0.14815693888248077
    ```
- **R2 Score on Test Set**: 
    ```
    0.9808019625757901
    ```
- **AIC Score**: 
    ```
    
    ```
### Visualization of Results

- A line chart visualizing the MSE and R-Squared scores for each **fold** to assess variability and performance across the folds:

![K-Fold Cross-Validation Results](<Images/K-Fold Cross-Validation Results.png>)

- A line chart visualizing the MSE and R-Squared scores to observe how the scores vary across **bootstrap samples**:

![Bootstrapping Results](<Images/Bootstrapping Results.png>)

- The Following plot is the Values of **MSE and RSquared Error** for the above two methods:

![Comparison of K-Fold and Bootstrapping](<Images/Average Scores Comparison.png>)





