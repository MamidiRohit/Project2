Ajay Anand A20581927 Anish Vishwanathan VR A20596106 Mohit Panchatcharam A20562455 Sibi Chandra sekar A20577946

1. What does the model you have implemented do and when should it be used?
2. How did you test your model to determine if it is working reasonably correctly?
3. What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

1.⁠ ⁠This algorithm predicts the health variables such as blood pressure, age, bmi and level of blood sugars to help determine if a patient may in
developing stage of diabetes. To identify those who are at serious level, it
searches for patterns in the data and operates in a manner that guarantees
effiecient and accurate results. How It Operates:
Preparing the data: To improve the models structure, the data is cleaned
up and modified. For example, to ensure consistency, figure out such as
glucose levels are in standard level. Examining the Model:
To test the model more than once, K Fold Testing divides the data into
smaller groups. This makes sure it works well on different sets of
information. Bootstrapping randomly samples the data to check how reliable the
model is over and over again. Improving Accuracy: The model’s settings are fine-tuned using a
method called Bayesian Optimization. This helps make the predictions as
accurate as possible. Understanding What Matters: The model identifies which health
factors, like glucose or age, are the most important for predicting diabetes. This makes it easier to see how the results were reached. When to Use:
This project is good for doctors, researchers, or hospitals. It can:
Help identify patients who might be at risk for diabetes early. Analyze health data to find common patterns or risk factors.

2.⁠ ⁠we tested the model to make sure it works properly and efficiently and
give the accurate results for the user to avoid any failures
TESTING STEPS:
Trying different data splits:
we split the data in several smaller groups using a technique called k
folding testing. the model was trained on some groups and test on others. This helped us to check if the model works properly how the data was
divided.We also used Bootstrapping where we randomly picked some
data samples and tested the model on the balance data. This results in
consistent results. Measuring accuracy:
We used scores like precision, recall, and AUC to see how the efficiently
the model was making a right decisions most of the time.The AUC ROC
curve showed us how the model was balanced between finding patients
with diabetes and avoiding false detection. Fine-Tuning for Better Results:
We adjusted the model’s settings to improve its accuracy. This step made
sure the predictions were as reliable and correct as possible. Making Sense of Results:
We checked which factors, like blood sugar levels or age, had the most
influence on the model’s predictions. This helped ensure the results
matched what we know about diabetes and made the model easier to trust.

3.This implementation allows the user to change some of the parameters
based on tuning the model performance. To specify how flexible the
model is for learning from data and predicting outputs. Here is a simple
guide to key parameters that the users can change:
Parameter Visibility:
n_estimators — Number of trees
It does this by controlling how many decision trees are present in the
model. Increased trees can result in better accuracy but can slow down
the model. For example: set it to 100 which means the model will use 100 trees for
predictions. Learning Rate (learning_rate):
This controls the weight of each tree in the final prediction. The model
tends to learn slower but often more accurately with smaller values (0.1)
Tree Depth (max_depth):
This defines how complex each tree could be. Larger trees can model
more features but they might also overfit to the training data. For example: if you set it to 5, at maximum each tree can consist of 5
levels. Minimum Samples to Split: (min_samples_split)
This sets threshold for the smallest amount of samples needed to split a
node in the tree. This model gets simpler as the value of becomes larger. For example, setting this value to 4 would require a minimum of 4
samples in order to perform a split. Minimum samples per leaf (min_samples_leaf):
This parameter controls the min samples at a leaf node. It helps prevent
overfitting. Example: The number of samples at leaf nodes must be greater than or
equal to this value, when set to 2 there will be a minimum of 2 sample in
the leaf node.
Subsampling (subsample):
This decides the fraction of data used to build each tree. Using less than
1.0 (e.g., 0.8) can reduce overfitting. Basic Example for Tuning:
Let’s say you want the model to train faster while maintaining good
accuracy. You can set:
n_estimators to 50
learning_rate to 0.1
max_depth to 3
By tweaking these parameters, you balance accuracy, speed, and how
well the model generalizes to new data.

4.Yes, the implementation may struggle with inputs that are highly imbalanced (e.g., one class is much smaller than the other) or have missing or noisy data. These issues can affect accuracy and fairness. With more time, we can address them by using techniques like oversampling, better preprocessing, or tweaking the model, so they aren't fundamental problems.

README

Overview:
In this project, we are building a machine learning model that helps predict whether a person has diabetes based on their health data. The dataset includes various health measurements, like glucose levels, age, BMI, and other factors. Our goal is to train a model that can accurately predict if someone is diabetic or not, based on this data.

In order to achieve the highest degree of accuracy in the model, we utilize various techniques which includes;
1.	Cross-Validation (Stratified K-Fold)
2.	Bootstrapping
3.	Hyperparameter Tuning (with Optuna)
4.	Feature Importance
5.	Model Evaluation (F1 Score, Precision, Recall, AUC)
Extra Techniques used in this project for bonus marks.

1. Bayesian optimization hyperparameter tuning.
2. K – fold base model.
 
Key Concepts and Techniques
1. Cross-Validation (Stratified K-Fold)
A method to test the model effectiveness by dividing the data into multiple subsets (or folds). This is stratified K-Fold Cross Validation and it is implemented such that number of diabetes-positive and negative cases are in a balanced proportion within each fold. This is useful to reliably estimate the performance of that model.
2. Bootstrapping
Bootstrapping is a statistical resampling method in which random samples are drawn from the data with replacement. We take a small subset of this data randomly and run the model on it couple of times, this way we can observe how the model behaves with different variations of the data. This gives us an indication of how the model might behave on completely new, never-before-seen before data.

3. Hyperparameter Tuning with Optuna
Hyperparameters are the settings in machine learning, by which a model learns. For instance, the number of trees in case of a decision tree model or a learning rate to control how fast the model learns by adjusting itself. Instead of going through all the parameters manually, we use a tool called optuna to automatically find the best hyperparameters for our model. This improves efficiency in model performance and saves us time.
4. Feature Importance
After training our model, we can see which factors (like age or glucose levels) contributed most to predicting if a person has diabetes. We call this process extraction of feature importance. By understanding which features are most important, we can learn what influences the prediction.
5. Evaluation Metrics
To measure how well our model is performing, we use several metrics:
•	F1 Score: This balances precision (how many of the predicted positives were actually positive) and recall (how many of the actual positives were correctly predicted).
•	Precision: The accuracy of the positive predictions.
•	Recall: How well the model finds the actual positive cases.
•	AUC (Area Under the Curve): This tells us how well the model can separate the two classes (diabetic vs. non-diabetic) based on probabilities.
 
Explanation of the Code
Step 1: Loading and Preparing the Data
First, we load the dataset, check its structure, and remove any unnecessary columns. We also check if any data is missing. After that, we standardize the data, which means we scale the features so they all have similar ranges, making it easier for the model to learn.
python
Copy code
# Loading the dataset
df = pd.read_csv("Healthcare-Diabetes.csv")
df.head()  # Show first few rows to understand data

# Checking if there are any missing values
df.isnull().any()  # There are no missing values

# Dropping 'Id' column as it's not needed for prediction
df.drop(columns=['Id'], inplace=True)
•	We remove the Id column because it's just a unique identifier and doesn't help in predicting diabetes.
•	We also check for missing data, and since there are none in this dataset, we don't have to worry about it.
Next, we standardize the features (like glucose levels, BMI, age, etc.) to make sure they all have the same scale. This helps the model perform better.
python
Copy code
# Standardize the numerical features
scaler = StandardScaler()
columns_to_standardize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
standardized_data = scaler.fit_transform(df[columns_to_standardize])

# Convert standardized data back into a DataFrame
df_standardized = pd.DataFrame(standardized_data, columns=columns_to_standardize)

# Add target variable (Outcome) back to standardized DataFrame
df_standardized["Outcome"] = df["Outcome"].values
•	We use StandardScaler to scale the data so that it has a mean of 0 and a standard deviation of 1. This helps the model treat all features equally, preventing some features from dominating due to their larger values.
 
Step 2: Training the Base Model Using Stratified K-Fold Cross-Validation
We use a machine learning model called Gradient Boosting Classifier. This model combines multiple decision trees to make predictions. We train and test the model using Stratified K-Fold Cross-Validation to get a more accurate measure of its performance.
python
Copy code
# Defining features and target
X = df_standardized.drop(columns=['Outcome'])  # Features
y = df_standardized['Outcome']                # Target variable

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)

# Initialize Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics
f1_scores_kfold = []
precision_scores_kfold = []
recall_scores_kfold = []
auc_scores_kfold = []
all_conf_matrices = []

for train_index, test_index in skf.split(X, y):
    # Splitting the data into train and test sets for each fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Training the model
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve
    
    # Calculate evaluation metrics
    f1_scores_kfold.append(f1_score(y_test, y_pred))
    precision_scores_kfold.append(precision_score(y_test, y_pred))
    recall_scores_kfold.append(recall_score(y_test, y_pred))
    auc_scores_kfold.append(roc_auc_score(y_test, y_pred_proba))
    
    # Save confusion matrix for later display
    conf_matrix = confusion_matrix(y_test, y_pred)
    all_conf_matrices.append(conf_matrix)

    # Plot ROC curve for this fold
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"K-Fold ROC Curve")

# Finalize and display the ROC curve
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("K-Fold AUC-ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Display confusion matrix for the last fold
disp = ConfusionMatrixDisplay(confusion_matrix=all_conf_matrices[-1], display_labels=model.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix for Last Fold in K-Fold Cross-Validation")
plt.show()

# Print out performance metrics
print(f"K-Fold F1 Scores: {f1_scores_kfold}")
print(f"K-Fold Precision Scores: {precision_scores_kfold}")
print(f"K-Fold Recall Scores: {recall_scores_kfold}")
print(f"K-Fold AUC Scores: {auc_scores_kfold}")
print(f"Mean F1 Score (K-Fold): {np.mean(f1_scores_kfold)}")
print(f"Mean AUC Score (K-Fold): {np.mean(auc_scores_kfold)}")
•	Stratified K-Fold means that the data is divided into 5 parts. Each part gets a chance to be used as both a training set and a testing set, making sure the data is balanced.
•	After training the model, we evaluate it using different metrics like F1 Score, Precision, Recall, and AUC. These metrics help us understand how well the model is performing.
We also plot an ROC curve to visually check how good the model is at distinguishing between diabetic and non-diabetic cases.
 
Step 3: Bootstrapping
Bootstrapping allows us to train the model on random subsets of the data, chosen with replacement. By repeating this process, we can get a better idea of how the model behaves with different portions of the dataset.
python
Copy code
# Initialize bootstrapping parameters
n_bootstrap_samples = 50
f1_scores_bootstrap = []
precision_scores_bootstrap = []
recall_scores_bootstrap = []
auc_scores_bootstrap = []

plt.figure(figsize=(8, 6))  # Set figure size for better visualization
confusion_matrices = []

for i in range(n_bootstrap_samples):
    # Create Bootstrap Sample
    indices = np.random.choice(range(len(X)), size=len(X), replace=True)
    X_train, y_train = X.iloc[indices], y.iloc[indices]
    
    # Train the model on the bootstrap sample
    model.fit(X_train, y_train)
    
    # Get out-of-bag (OOB) predictions
    oob_indices = list(set(range(len(X))) - set(indices))
    X_test_oob, y_test_oob = X.iloc[oob_indices], y.iloc[oob_indices]
    
    # Predict using the model
    y_pred = model.predict(X_test_oob)
    y_pred_proba = model.predict_proba(X_test_oob)[:, 1]  # Probabilities for ROC curve
    
    # Collect performance metrics
    f1_scores_bootstrap.append(f1_score(y_test_oob, y_pred))
    precision_scores_bootstrap.append(precision_score(y_test_oob, y_pred))
    recall_scores_bootstrap.append(recall_score(y_test_oob, y_pred))
    auc_scores_bootstrap.append(roc_auc_score(y_test_oob, y_pred_proba))
    
    # Plot ROC Curve for each bootstrap sample
    fpr, tpr, _ = roc_curve(y_test_oob, y_pred_proba)
    plt.plot(fpr, tpr, label=f"Sample {i+1}")

# Finalize and plot the ROC curves for bootstrapping
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Bootstrapping AUC-ROC Curve")
plt.legend(loc="lower right")
plt.show()
•	We train the model on 50 bootstrapped samples, calculate evaluation metrics for each sample, and plot the ROC curve for each of them. This helps us understand how the model performs across different subsets of data.
 
Step 4: Hyperparameter Tuning with Optuna
To make the model even better, we use Optuna to automatically search for the best model settings, called hyperparameters. These include things like how many trees to use, what the learning rate should be, and more.
python
Copy code
import optuna

# Define the objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = GradientBoostingClassifier(random_state=42, **params)
    scores = cross_val_score(model, X, y, scoring='f1', cv=3, n_jobs=-1)
    return scores.mean()

# Start Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
•	Optuna helps us find the best hyperparameters by trying different combinations and evaluating their performance automatically.
 
Conclusion
By using techniques like cross-validation, bootstrapping, and hyperparameter tuning, we've made sure that our diabetes prediction model isn't just accurate, but also reliable when faced with new data it hasn't seen before. These methods help strengthen the model, making it more flexible and capable of providing better predictions in real-world scenarios. With each step, we've worked to improve how well the model can understand and predict outcomes, ensuring it's both trustworthy and useful.
