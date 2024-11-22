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


Put your README below. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?


