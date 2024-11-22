***Gradient Boosting Trees and Model Selection***

***GROUP NAME: GUARDIANS OF IIT***

***TEAMMEMBERS***
##### A20584318 - ANSH KAUSHIK
##### A20593046 - ARUNESHWARAN SIVAKUMAR
##### A20588339 - HARISH NAMASIVAYAM MUTHUSWAMY
##### A20579993 - SHARANYA MISHRA
##### A20541092 - SHUBHAM BAJIRAO DHANAVADE 

***CONTRIBUTION***
We, as a team of five, collaboratively worked on the implementation, testing, and documentation of the Gradient Boosting Tree model and the associated model selection methods. Each team member contributed equally, taking responsibility for specific aspects of the project to ensure its successful completion.

**CS584 Project 2:**
**Gradient Boosting and Model Selection**

This repository contains two independent projects:

Gradient Boosting: Implements and evaluates the Gradient Boosting algorithm for classification and regression tasks.
Model Selection: Focuses on evaluating and comparing machine learning models using robust methods like bootstrapping and cross-validation.

*********************************************************1. Gradient Boosting******************************************************
**Purpose:**
The Gradient Boosting project focuses on implementing and evaluating the Gradient Boosting algorithm, a popular ensemble method for improving model accuracy by sequentially reducing prediction errors.

**Project Structure (Gradient Boosting):**
    1. main_gradientboosting.py: The primary script to execute Gradient Boosting.
    2. boosting/: folder containing Gradient boosting model
            gradient_boosting.py: Initialization of Model for Gradient Boosting.
    3. comparison-gbt.py: This shows the comparison between sk learn model and our model.
    4. Datasets: Located in the data/ directory:
            highly_correlated_dataset.csv and partially_correlated_dataset.csv:
                These are synthetic datasets generated from code, designed to test the impact of feature correlation on model performance.
            boston_housing.csv, highly_correlated_dataset.csv, partially_correlated_dataset 
    5. comparison-gbt.py - the comaprison file for our model and sk learn
**Setup for Gradient Boosting:**
1. Clone the Repository:
        git clone <"https://github.com/mishrasharanya/CS584_Project2.git">  
        cd CS584_Project2-main  

2. Install Dependencies:
    Ensure Python 3.11.7 is installed, then run:
        pip install -r requirements.txt  

3. Verify Data:
Ensure datasets are present in the data/ directory. If missing, use a compatible dataset in CSV format.

**How to Run Gradient Boosting**
1. Execute the Script:
    Run the Gradient Boosting pipeline:
        test_boosting.py - this is a test file to test if its working.
        main_gradientboosting.py - This is the main file to be run to get the output.
        comparison-gbt.py - this is the file to comapare with sk learn with our model.


Output:
    Model training progress and evaluation metrics (MSE, MAE, R2).
    Visualizations (Actual Vs predicted Graph) saved to the output directory.

**Customization for Gradient Boosting**
    Hyperparameters: Modify parameters like the learning rate, number of iterations, or max depth directly in Gradient_Boosting.py.
    Dataset: Replace the default dataset (data/train.csv) with a custom dataset by updating main_gradientboosting.py.


*************************************************Gradient BOOSTING QUESTIONS******************************************************

1. What Does the Model You Have Implemented Do and When Should It Be Used?
The Gradient Boosting Tree model is a powerful supervised learning algorithm for regression tasks. It builds an ensemble of weak learners (shallow decision trees) by iteratively training each tree to correct the residual errors of the previous ones. This sequential approach allows the model to capture complex, non-linear relationships in data. The final prediction combines the outputs of all trees, scaled by a learning rate to balance learning and generalization.

Ideal for structured, tabular data, Gradient Boosting excels in tasks requiring high predictive accuracy, such as finance, healthcare, and marketing. Despite being less interpretable than simpler models, it is highly effective when model performance is prioritized. Careful tuning of hyperparameters like learning rate, tree depth, and the number of trees is crucial to prevent overfitting and optimize results, making this model a versatile choice for complex regression problems.


2. How Did You Test Your Model to Determine if It Is Working Reasonably Correctly?
Gradient Boosting Tree Model Testing Overview

The model was rigorously tested using synthetic and real-world datasets to ensure reliability and accuracy.

**Datasets Used:**
Synthetic: Small datasets (Synthetic generated Datasets namely partially correlated and highly correlated) tested basic functionality and logic correctness.
Real-World: Benchmarks like boston_housing.csv assessed generalization and practical performance.

**Evaluation Metrics:**
Mean Squared Error (MSE): Measures overall accuracy by averaging squared errors.
Mean Absolute Error (MAE): Captures average prediction deviation in interpretable units.
R² Score: Quantifies explained variance; closer to 1 indicates a better fit.

**Testing Methods:**
Unit Tests: Validated model functions (fit, predict) with predefined data.
Cross-Validation: Ensured consistency across multiple data splits.
Ground Truth Comparison: Verified prediction alignment with actual target values.
Key tests confirmed minimized residuals, strong predictive accuracy, and robust generalization, combining controlled experiments and practical scenarios for reliability.

3. What Parameters Have You Exposed to Users of Your Implementation to Tune Performance?
**Optimizing Gradient Boosting Tree Performance**

The Gradient Boosting Tree model offers key parameters for tuning to balance accuracy and generalization:

n_estimators (Number of Trees)

Determines the number of boosting iterations.
Higher values improve performance but can lead to overfitting if unchecked.
Start with 100 and adjust based on validation performance.
learning_rate (Step Size)

Scales each tree's contribution to the final prediction.
Smaller values (e.g., 0.01–0.1) improve generalization but require more trees.
Larger values risk overfitting.
max_depth (Tree Depth)

Limits tree complexity to control overfitting.
Deeper trees capture more patterns but may memorize noise.
Start with 2 and increase only if underfitting occurs.
Tuning Tips

Begin with default values: n_estimators=100, learning_rate=0.1, max_depth=2.
Tune sequentially: Adjust n_estimators, lower learning_rate, and tweak max_depth.
Use cross-validation and grid or random search for robust tuning.
By carefully optimizing these parameters, the model can adapt to various datasets and tasks, maximizing accuracy while avoiding overfitting.

4. Are There Specific Inputs That Your Implementation Has Trouble With?
**Challenges and Future Improvements for Gradient Boosting Trees**
**Challenges**
**High-Dimensional Sparse Data**
Issue: Slower computation and risk of overfitting due to irrelevant features.
Outliers
Issue: Sensitivity to extreme values can lead to overfitting.
Highly Correlated Features

Issue: Redundant splits increase complexity without adding value.


**Regularization**
Add L1/L2 penalties to reduce overfitting and improve generalization by limiting tree complexity.
Early Stopping
Halt training when validation performance plateaus, preventing overfitting and saving resources.
Classification Support
Extend the model for binary and multiclass tasks using log-loss or softmax loss, enabling applications in fraud detection and sentiment analysis.
These enhancements ensure robust, efficient, and versatile performance across diverse datasets and tasks.

*****************************************************2. Model Selection***********************************************************
**Purpose**
The Model Selection project demonstrates robust techniques for evaluating and selecting the best machine learning models. It uses bootstrapping and cross-validation to ensure reliable model performance comparisons.

**Project Structure (Model Selection):**
    1. main_modelselection.py: The main script for performing model selection.
    2. model_selection/: folder containing boot strapping and cross validation.
        bootstrapping.py: Implements bootstrapping for model evaluation.
        cross_validation.py: Cross-validation utilities for splitting datasets.
    3. Datasets: Located in the data/ directory:
        highly_correlated_dataset.csv and partially_correlated_dataset.csv:
            These are synthetic datasets generated from code, designed to test the impact of feature correlation on model performance.
        Boston_housing.csv: A real-world dataset.

**Setup for Model Selection:**
 
1. Install Dependencies:
Install required Python libraries:
    pip install -r requirements.txt  
2. Verify Data:
    Ensure datasets are present in the data/ directory. Synthetic datasets (highly_correlated_dataset.csv and partially_correlated_dataset.csv) are pre-generated and included. If needed, refer to the provided code to regenerate these datasets.

**How to Run Model Selection**
1. Execute the Script:
        Run the model selection pipeline:
            test_modelselection.py - this is a test file to test if its working.
            main_modelselection.py - This is the main file to be run to get the output.
            comparison-modelselection.py - this is a comparison file betweern our model and SK learn
Output:
    Performance metrics (K- Fold MSE, K Fold Crossvalidation, Bootstrap Mean MSE, Bootstrap Scores, Max Depth- 3 nos CV MSE-3 nos and Best max Depth And Best CV MSE) 
    Summary of the best-performing model.

**Customization for Model Selection**
    1. Evaluation Method:
        Choose between bootstrapping or cross-validation in main_modelselection.py.
    2. Hyperparameters:
        Update cross-validation folds or bootstrapping parameters in cross_validation.py or bootstrapping.py.
    3. Models:
        Add or modify models in ModelSelection.py for comparison.

*****************************************************MODEL SELECTION QUESTIONS**********************************************
1. Do Your Cross-Validation and Bootstrapping Model Selectors Agree with a Simpler Model Selector Like AIC?

**Agreement Between Model Selectors**

**Simple Models (e.g., Linear Regression):**
Techniques like AIC, cross-validation, and bootstrapping often agree.
Why AIC Works: Assumes Gaussian residuals and penalizes complexity, aligning well with linear regression's assumptions.
Cross-Validation/Bootstrapping: Evaluate on unseen or resampled data, reinforcing simpler models, similar to AIC.

**Complex Models (e.g., Gradient Boosting Trees):**
Cross-validation and bootstrapping are more reliable than AIC.
Why AIC Falls Short: Assumes parametric structure and Gaussian residuals, which do not apply to non-linear, non-parametric models.

**Advantages of Cross-Validation/Bootstrapping:**
Cross-Validation: Directly measures generalization on unseen data.
Bootstrapping: Tests stability and variance through resampling.

These methods adapt to flexible models like Gradient Boosting Trees, ensuring robust and reliable model selection without reliance on strict parametric assumptions.

2. In What Cases Might the Methods You've Written Fail or Give Incorrect or Undesirable Results?

**Challenges in Model Selection and Mitigation Strategies**

**Small Datasets**

Challenge: High variance in cross-validation; bootstrapping lacks diversity.
Mitigation: Use stratified sampling to preserve target distribution and consider leave-one-out cross-validation (LOOCV) for reliable estimates.

**Imbalanced Datasets**

Challenge: Cross-validation may not maintain class proportions, leading to biased metrics.
Mitigation: Apply stratified k-fold cross-validation to maintain class balance and use oversampling techniques like SMOTE for minority classes.

**Outliers**

Challenge: Outliers distort evaluation and cause overfitting.
Mitigation:  normalize features

**Hyperparameter Search with Large Grids**

Challenge: Large grids lead to high computational costs.
Mitigation: Use randomized search for faster exploration, focus on impactful parameters (e.g., learning_rate), and parallelize the search with multiprocessing or distributed frameworks.
By addressing these challenges with targeted strategies, model selection can be made more efficient and reliable.


3. What Could You Implement Given More Time to Mitigate These Cases or Help Users of Your Methods?
**Future Enhancements for Model Selection and Training**

**Stratified Sampling**

Challenge: Imbalanced datasets may lead to biased evaluation in cross-validation.
Solution: Use stratified K-fold cross-validation to maintain class proportions in all folds.
Benefit: Provides reliable metrics, especially for minority classes.
Weighted Bootstrapping

Challenge: Standard bootstrapping over-samples outliers or misses minority classes.
Solution: Assign higher weights to minority samples and lower weights to outliers during resampling.
Benefit: Improves balance and reduces noise impact.
Early Stopping in Grid Search

Challenge: Large search spaces make grid search computationally expensive.
Solution: Stop the search when performance stops improving over a set number of iterations.
Benefit: Saves time and focuses resources on promising parameters.
Parallelization

Challenge: Cross-validation and grid search are time-intensive.
Solution: Use parallel processing to distribute tasks across multiple cores or nodes.
Benefit: Significantly reduces computation time.
Feature Selection in Cross-Validation

Challenge: Irrelevant features increase overfitting and computational cost.
Solution: Apply automated feature selection during cross-validation to retain only relevant features.
Benefit: Improves generalization, interpretability, and efficiency.
These strategies enhance the efficiency, accuracy, and robustness of model selection and training processes.

4. What Parameters Have You Exposed to Your Users in Order to Use Your Model Selectors?
Key Parameters for Model Selection

K-Fold Cross-Validation

k: Number of folds. Higher k (e.g., 10) gives robust estimates but is computationally expensive.
shuffle & random_state: Ensures randomness and reproducibility during data splits.
metric: Determines performance evaluation (e.g., MSE for regression).
Bootstrapping

B: Number of resampling iterations. Higher B improves variability estimates but increases computation.
metric: Measures performance (e.g., MSE, MAE).
Weighted Sampling: Enhances robustness for imbalanced data.
Grid Search

param_grid: Defines hyperparameter values to explore. Larger grids improve results but increase runtime.
Early Stopping: Halts search when performance stabilizes, saving time.
Parallelization: Speeds up grid search by processing multiple combinations simultaneously.
Strengths

Suitable for all model types, including non-parametric models.
Cross-validation ensures generalization, while bootstrapping assesses variability.
Challenges


Future Enhancements:

Stratified Sampling: Maintains class balance in folds.
Weighted Bootstrapping: Handles imbalances and outliers effectively.
Automated Feature Selection: Improves generalization by reducing noise.
Parallelization: Cuts runtime for large-scale tasks.
These adjustments improve the efficiency, accuracy, and scalability of model selection processes.

**Notes:**
Synthetic Datasets:
    1. highly_correlated_dataset.csv: A synthetic dataset where features have high inter-correlation.
    2. partially_correlated_dataset.csv: A synthetic dataset with moderate feature correlation.
    3. These datasets were generated using custom Python code, which can be found in the project.
    4. Independent Projects: Each project can be run independently. Follow the respective setup and execution instructions for Gradient Boosting or Model Selection.
    5. Extensibility: New models or datasets can be added easily. Update the relevant scripts and configurations as needed.
    6. Troubleshooting: Run unit tests or verify dataset formats if errors occur.

**Key Takeaways**
**Non-Linear Model Evaluation:**
Cross-validation and bootstrapping are indispensable for evaluating complex models like Gradient Boosting Trees, as they empirically assess performance without relying on restrictive assumptions.
Alignment with Simpler Methods:
While these methods align with simpler selectors like AIC for linear models, their flexibility makes them more suitable for modern machine learning tasks.
Enhancements for Scalability:
By incorporating stratified sampling, parallelization, and other optimizations, these methods can become even more efficient and effective for large-scale or computationally demanding applications.


*******

***Conclusion***
Cross-validation and bootstrapping provide a robust foundation for model evaluation and selection, particularly for complex tasks involving non-linear relationships. Their empirical, assumption-free approach ensures reliable performance estimation and generalization, even for cutting-edge machine learning models. With future enhancements like parallelization and automated feature selection, these techniques can be scaled and optimized to meet the growing demands of modern data science.










