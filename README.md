# Machine Learning
## Project 2 

# Group Members - Contribution

* Venkata Naga Lakshmi Sai Snigdha Sri Jata - A20560684 - 33.33%
* Sharan Rama Prakash Shenoy - A20560684 - 33.33%
* Adarsh Chidirala - A20561069 - 33.33%

# ###################################################
## Usage Instructions

### Installation

To get started with this project, first you need **Python 3.x**. Then follow these installation steps:

#### 1. Clone the Repository to your local machine:

```bash
git clone https://github.com/adarsh-chidirala/Project2_Adarsh_Ch_Group.git 
```
#### 2. Steps to Run the Code on Mac

Follow these steps to set up and run the project:

1. **Create a Virtual Environment**:
   - Navigate to your project directory and create a virtual environment using:
     ```bash
     python3 -m venv myenv
     ```

2. **Activate the Virtual Environment**:
   - Activate the created virtual environment by running:
     ```bash
     source myenv/bin/activate
     ```

3. **Install Required Libraries**:
   - Install the necessary Python libraries with the following command:
     ```bash
     pip install numpy pandas matplotlib scikit-learn
     ```

4. **Run the Script**:
   - Navigate to the directory containing your script and run it:
     ```bash
     python project2.py
     ```

Make sure that the script `project2.py` and any required dataset files are correctly placed in your project directory.

#### 3. Steps to Run the Code on Windows

Follow these instructions to set up and execute the project on a Windows system:

1. **Create a Virtual Environment**:
   - Open Command Prompt and navigate to your project directory:
     ```cmd
     cd path\to\your\project\directory
     ```
   - Create a virtual environment in your project directory by running:
     ```cmd
     python -m venv myenv
     ```

2. **Activate the Virtual Environment**:
   - Activate the virtual environment with the following command:
     ```cmd
     myenv\Scripts\activate
     ```

3. **Install Required Libraries**:
   - Install the necessary libraries by executing:
     ```cmd
     pip install numpy pandas matplotlib scikit-learn
     ```

4. **Run the Script**:
   - Make sure the script `project2.py` and any necessary dataset files are placed in your project directory. Run the script with:
     ```cmd
     python project2.py
     ```
Ensure that all paths are correct and relevant files are located in the specified directories.

#### 4. Running Datasets:
   - We are using two datasets: customer_dataset.csv and patient_data.csv. For each of this we need to specify the features and the target column. 
   - This is done in the main function in the bottom of the code as follows.
   - For customer_dataset.csv
``` 
data = pd.read_csv('customer_dataset.csv')

    feature_columns = ['Age','Annual_Income','Spending_Score']
    target_column = 'Store_Visits'

```
   - For patient_data.csv
``` 
data = pd.read_csv('patient_data.csv')

    feature_columns = ['RR_Interval','QRS_Duration','QT_Interval']
    target_column = 'Heart_Rate'

```

## K-fold cross-validation and Bootstrapping model selection models
## Introduction
- This project implements the developments of generic k-fold cross-validation and bootstrapping model selection methods, primarily focusing on AIC scores to evaluate the model performance. 
- These techniques are designed to evaluate and compare the performance of machine learning models on different datasets, providing detailed insights into how effective a model is on the given situations and predict the corresponding outcomes accurately based on the scores calculated. 
- The implementation can be adapted to various models, and allows customization to suit individual requirements and resources by modifying its features.

## Key Features implemented
- The model supports 3 different models i.e; Linear, Ridge and Lasso regression models and calculates Mean AIC scores for each model using following validation models.
- The model successfully implements K-Fold Cross-Validation and Bootstrapping Validation.
- **K-Fold Cross-Validation:** This validation model provides evaluation by splitting the dataset into k subsets, using k-1 subsets for training the dataset and remaining 1 subset for testing the dataset.
- **Bootstrapping:** This validation model provides evaluation of the performance by repeatedly resampling the dataset with replacements of the same data.
- The project also generates the plots of the K-fold cross validation and Bootstrapping AIC scores and Distribution.


### 1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
In simple cases like linear regression,the relationship between the dependent and independent variables is considered to be linear, it is possible that cross-validation, bootstrapping, and AIC scores may accept on the simpler model selector. 
However, this may not be constant always because it depends on the dataset provided and the complexity of the models.

Note that cross-validation and bootstrapping provide estimates of model performance, while AIC focuses on model selection. These techniques can be different in model evaluation and selection based on the different purposes.

To determine whether cross-validation and bootstrapping model selectors agree with AIC in a specific case, it is recommended to perform experiments and compare the results.

### 2. In what cases might the methods you've written fail or give incorrect or undesirable results?
Failures or undesirable results may occur under these conditions:
**Small Datasets:** Cross-validation and bootstrapping can give unreliable AIC scores when the dataset is too small.
**Non-balance Data:** Models trained on non-balance datasets might make biased predictions which affects the AIC calculations.
**Connected Predictions:** Strong connections between predictions can lead to wrong estimates, affecting AIC evaluations.
**Wrong Model Assumptions:** AIC relies on specific model assumptions (e.g., linear regression assumes Gaussian errors). If these assumptions are wrong, AIC might not show the true model fit.
**Overfitting with Bootstrapping:** Repeated sampling with replacement might make bootstrapped datasets favor complex models too much, leading to high AIC values.

### 3. What could you implement given more time to mitigate these cases or help users of your methods?

1. Use Ridge and Lasso penalties during cross-validation: Regularization adds a penalty to the model's complexity to reduce overfitting. Ridge regression adds an L2 penalty (squared magnitude_value  of coefficients), while Lasso regression adds an L1 penalty (absolute value of coefficients).
Use RidgeCV and LassoCV from scikit-learn: These classes perform cross-validation to find the better regularization parameter (alpha) for Ridge and Lasso regression.
Handle Imbalanced Data:

2. Use categorized sampling for k-fold cross-validation: Categorized k-fold ensures that each fold has the equal proportion of class labels as the original dataset, which shows imbalanced datasets.
Implement balanced bootstrapping: Bootstrapping involves sampling with a stategy of replacement. Balanced bootstrapping ensures that each class is represented equally in each sample.

3. Add BIC or deviance as options besides AIC: AIC score is a common metric for model selection, but BIC (Bayesian Information Criterion) and deviance can provide additional insights. 

4. Plot performance for different hyperparameters: Validation curves help visualize how the model's performance changes with different values of hyperparameters (e.g., alpha in Ridge/Lasso). This helps in selecting the correct hyperparameter value.

5. Provide summaries for bootstrap and k-fold iterations: Summarizing the results of resampling methods (like bootstrap and k-fold) helps understand the variability and stability of the model's performance.

6. Let users set evaluation metrics and sampling methods: Providing flexibility in setting these parameters allows users to customize the model according to their specific needs and preferences.

### 4. What parameters have you exposed to your users in order to use your model selectors.
1. **Cross-Validation:**
k: Number of parts to split the data into for testing.
model_type: Type of model (e.g., 'linear', 'ridge', 'lasso', 'logistic').
2. **Bootstrapping:**
n_bootstraps: Number of times to randomly sample the data.
model_type: Type of model (e.g., 'linear', 'ridge', 'lasso', 'logistic').
3. **General Settings:**
Features (X_columns) and target (y_column): Columns to use from any dataset.


### Code Visualization:
- The following screenshots display the results of each test case implemented in this project:

### 1. customer_dataset.csv:
- Tests the model on a small dataset, and verifies if the predictions are reasonable.
- i. K-fold cross validation AIC score:
    ![Customer Test Image](customer_aic_k_fold.jpeg)
- ii. Bootstrap AIC score:
    ![Customer Test Image](customer_aic_bootstrap.jpeg)
- iii. K-fold cross validation AIC distribution:
    ![Customer Test Image](customer_ aic_distr_k_fold.jpeg)
- iv. Bootstrap AIC distribution:
    ![Customer Test Image](customer_ aic_dis_bootstrap.jpeg)
- v. Bootstrap Output:
    ![Customer Test Image](customer_output.jpeg)

### 2. patient_data.csv:
- Tests the model on a large dataset, and verifies if the predictions are reasonable.
- i. K-fold cross validation AIC score:
    ![Patient Test Image](patient_ aic_k_fold.jpeg)
- ii. Bootstrap AIC score:
    ![Patient Test Image](patient_ aic_bootstrap.jpeg)
- iii. K-fold cross validation AIC distribution:
    ![Patient Test Image](patient_ aic_dis_k_fold.jpeg)
- iv. Bootstrap AIC distribution:
    ![Patient Test Image](patient_ aic_dis_bootstrap.jpeg)
- v. Bootstrap AIC distribution:
    ![Patient Test Image](patient_output.jpeg)
