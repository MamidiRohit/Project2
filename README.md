# Project 2

Gradient Boosting Tree Implementation Analysis

## TEAM MEMBERS
1. Munish Patel - mpatel176@hawk.iit.edu (A20544034)
2. Jaya Karthik Muppinidi - jmuppinidi@hawk.iit.edu (A20551726)
3. Meghana Mahamkali - mmahamkali@hawk.iit.edu (A20564182)
4. Nirusha Mantralaya Ramesh - nmantralayaramesh@hawk.iit.edu (A20600814)

## CONTRIBUTIONS
1. ALGORITHM DEVELOPMENT AND DATA PROCESSING (MUNISH,NIRUSHA)
2. MODEL IMPLIMENTATION AND TESTING FRAMEWORK(MEGHANA,KARTHIK)

### Initial Data Used
  https://github.com/munishpatel/ML-DATA/blob/c9442334645ca2ac71820578d17125c630c6199f/mldata.csv

## REQUIREMENTS
PYTHON 3.7+
NUMPY
SCKIT

## PROJECT OVERVIEW 

Core Components
1. GradientBoostingTreeRegressor Class
This is the main class that implements gradient boosting for regression tasks. 
Key features include:
Initialization Parameters:
* n_estimators: Number of boosting stages (default=100)
* learning_rate: Step size for gradient descent (default=0.1)
* max_depth: Maximum depth of each decision tree (default=3)
* random_state: Random seed for reproducibility
Key Methods:
* _initialize_f0: Initializes the baseline prediction as mean of target values
* _negative_gradient: Computes residuals (y - prediction) as negative gradient
* fit: Trains the model using sequential boosting
* predict: Generates predictions by combining all trees
2. Early Stopping Mechanism
The implementation includes an intelligent early stopping feature:
* Uses patience parameter (500 iterations)
* Monitors MSE for improvement
* Stops if no improvement seen after patience iterations
* Helps prevent overfitting and reduces unnecessary computations
3. Data Handling (load_data function)
Sophisticated data preprocessing with:
* Automatic handling of both numeric and categorical features
* LabelEncoder for categorical variables
* Robust error handling for data conversion
4. Model Testing Framework (test_model function)
Comprehensive testing suite including:
* Data scaling using MinMaxScaler
* Model training with specified parameters
* Performance metrics calculation (MSE and R² Score)
* Visualization of predictions vs actuals

MAIN Features
1. Adaptive Learning
* Uses residual-based learning where each tree corrects previous trees' errors
* Learning rate controls the contribution of each tree
* Combines weak learners (decision trees) into a strong predictor
2. Preprocessing
* Automatic feature type detection
* Scales target values to [0,1] range
* Handles missing data and categorical variables
3. Visualization
* Scatter plot of predicted vs actual values
* Reference line for perfect predictions
* Grid for better readability

Implementation Strengths
1. Robustness:
    * Handles different data types automatically
    * Includes error checking and data validation
    * Uses early stopping to prevent overfitting
2. Flexibility:
    * Customizable hyperparameters
    * Compatible with scikit-learn API
    * Can handle both regression and classification tasks (through scaling)
3. Performance Optimization:
    * Early stopping reduces unnecessary computations
    * Uses numpy for efficient array operations
    * Intelligent data preprocessing

Technical Details
Key Algorithms
1. Base Prediction:
f0 = mean(y)
2. Gradient Calculation:
residuals = y - current_predictions
3. Model Update:
predictions += learning_rate * tree.predict(X)
Implementation Notes
* Uses scikit-learn's DecisionTreeRegressor as base learner
* Inherits from BaseEstimator and RegressorMixin for compatibility
* Implements numpy vectorization for efficiency
Performance Characteristics
* Time Complexity: O(n_estimators * n * log(n)) where n is number of samples
* Space Complexity: O(n_estimators * tree_size)
* Early stopping can significantly reduce actual runtime


## Answers to the following questions:
1. What does the model you have implemented do and when should it be used?

  - Gradient Boosting Regressor model is implemented that predict continuous, numerical values using successive decision trees constructed to minimize residual errors. 
  - The negative gradient of the loss function is used to train each tree so that prediction get better over iterations. 
  - For tasks with complex, non-linear relationships on the data, the models are especially effective for regression tasks.
  - Also, these parameters provide fine grain bias variance tradeoffs for control over model complexity and learning speed: tree depth, learning rate, and no. of estimators. 
  - The models have found good application for tasks such as forecasting, pricing optimization and other predictive analytics where high accuracy is key.

2. How did you test your model to determine if it is working reasonably correctly?
   
  - We evaluated our model by training it on a dataset that predicts suggested job roles.
  - The models were evaluated on provided datasets by means of metrics such as Mean Squared Error (MSE) and R2 score to quantify prediction accuracy.
  - Scaling using MinMaxScaler or RobustScaler was applied as preprocessing technique to allow compatibility with the model, and to increase performance on the test data.
  - Additionally the implementations also had an early stopping mechanism to avoid overfitting by stopping training when improvements reduced.
  - With verification these testing methods confirmed that the models yield robust and accurate predictions where the dataset lies.

3. What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
   
    Exposed Parameters:
  - ```n_estimators```: Number of boosting iterations.
  - ```learning_rate```: Step size for updating predictions.
  - ```max_depth```: Maximum depth of individual trees.
  - ```random_state```: Seed for reproducibility.

    Example usage:
  -```
      model = GradientBoostingTreeRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test) ```
    
4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

  - Challenges:
    * This can be used to handle categorical data without preprocessing (yes, it can be done but with a simple step of label encoding).
    * As shown for example with scaling using RobustScaler or MinMaxScaler, not performing well on noisy datasets or outliers without robust preprocessing.
  - Workarounds:
    * Naturally extend the implementation to support categorical features (one hot encoding).
    * Integrate additional regularization techniques or automated outlier detection that generates further robustness to noise.
    * These are not fundamental problems, if further development can address them.

## Steps to run the code

1. **Set Up the Environment**:
   - Make sure you have the data .csv files in the same directory as the .ipynb file.
   - Install dependencies with the following command:
     ```bash
     pip install numpy matplotlib scikit-learn joblib scipy
     ```
   - Ensure you are using **Python 3.8+** and running the code in **Google Colab**.

2. **Execution**
   - Clone the existing repository from the github or download the python script.
   - Open the script in the Google Colab, you can achieve this by copy pasting the content or you can also directly upload it in Colab interface.
   - Run/Execute the code.
  
