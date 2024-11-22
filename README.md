### **Model Selection using K-Fold Cross-Validation and Bootstrap**

---

### **Project Overview**

This project evaluates machine learning models using **K-Fold Cross-Validation** and **Bootstrap Resampling**. It aims to select the best model based on predictive performance and robustness. The evaluation is performed on the **Digits dataset** using hyperparameter tuning for optimal results.

The following classifiers are included:
- Random Forest
- Logistic Regression
- Support Vector Machines (SVM)
- K-Neighbors Classifier
- Decision Tree

Logs, reports, and visualizations provide detailed insights into model performance.

---

### **How to Run the Code**

#### **Prerequisites**
1. **Python Version**: Ensure Python 3.10 or higher is installed.
2. **Dependencies**: Install required libraries with:
   ```bash
   pip install -r requirements.txt
   ```

#### **Execution Steps**
1. Clone the repository or download the Python script.
2. Navigate to the script directory.
3. Run the script using:
   ```bash
   python Project2_KCross_Bootstrap.py
   ```

#### **Outputs**
- Logs: Saved in `logs/debug.log`.
- Reports: Stored in `results/`.
- Visualizations: Saved as `.png` files in `figures/`.

---

### **Answers to Key Questions**

#### **1. Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?**

Yes, in simpler cases such as linear regression, both **K-Fold Cross-Validation** and **Bootstrap Resampling** align well with simpler model selectors like **AIC**. These methods focus on evaluating model performance, balancing complexity and predictive accuracy:
- **K-Fold Cross-Validation** evaluates models across multiple splits, providing stable estimates of performance variability.
- **Bootstrap Resampling** assesses generalization by evaluating predictions on resampled data.

While AIC relies on assumptions like model linearity, K-Fold and Bootstrap are more versatile, making them suitable for evaluating a wider range of models.

#### **2. In what cases might the methods you've written fail or give incorrect or undesirable results?**

The methods may face challenges in the following cases:
- **Imbalanced Datasets**: K-Fold might fail to preserve class distributions across splits, leading to misleading results.
- **Small Datasets**: Both methods can struggle with small datasets. K-Fold may lose critical data in splits, while Bootstrap may produce overly optimistic estimates.
- **Overfitting During Tuning**: Excessive hyperparameter tuning in K-Fold can lead to overfitting, causing poor performance on unseen data.

Such issues can result in models being incorrectly evaluated, favoring those that perform well on the validation data but fail on unseen data.

#### **3. What could you implement given more time to mitigate these cases or help users of your methods?**

With more time, the following improvements could be implemented:
1. **Stratified K-Fold**: Automatically preserve class distributions across folds for imbalanced datasets.
2. **Nested Cross-Validation**: Separate hyperparameter tuning and evaluation to avoid overfitting and provide a true estimate of model performance.
3. **Advanced Bootstrap Adjustments**:
   - `.632+ Bootstrap` for more realistic error estimation.
   - Custom sampling ratios for flexible resampling.
4. **Additional Metrics**: Provide F1-score, ROC-AUC, or precision-recall curves for a more nuanced evaluation.
5. **Automated Error Analysis**: Generate reports highlighting misclassification patterns and critical features.

#### **4. What parameters have you exposed to your users in order to use your model selectors?**

The program provides flexibility by exposing the following parameters:
- **K-Fold Parameters**:
  - Number of splits (`n_splits_values`): Users can specify split sizes, such as 5 or 10.
- **Bootstrap Parameters**:
  - Number of iterations (`n_iterations`): Control the resampling count (default: 100).
- **Model Hyperparameters**:
  - Random Forest: `n_estimators` (number of trees).
  - Logistic Regression: Regularization parameter `C`.
  - SVM: Regularization parameter `C` and kernel coefficient `gamma`.
  - Decision Tree: Maximum depth (`max_depth`).
  - K-Neighbors: Number of neighbors (`n_neighbors`).
- **Output Controls**:
  - Logs: Debug information and runtime statistics saved in `logs/`.
  - Reports: Summaries of metrics and hyperparameters stored in `results/`.

---

### **Expected Outputs**

1. **Logs**:
   - Debug and runtime logs are saved to `logs/debug.log`.

2. **Reports**:
   - K-Fold results with metrics and hyperparameters: `results/kfold_results.txt`.
   - Bootstrap results with mean and standard deviation: `results/bootstrap_results.txt`.

3. **Visualizations**:
   - **K-Fold Visualizations**:
     - Mean ± Standard Deviation of scores across models.
     - Scores across different fold sizes.
   - **Bootstrap Visualizations**:
     - Trends of accuracy over iterations.
     - Distribution of bootstrap scores.

All figures are saved as `.png` files in the `figures/` directory.

---

### **Summary**

This program provides robust model evaluation using K-Fold Cross-Validation and Bootstrap Resampling. It ensures flexibility for various datasets and offers detailed logs, reports, and visualizations, helping users make informed model selection decisions.
