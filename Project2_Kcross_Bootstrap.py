import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# Create folders for results, figures, and logs
results_dir = "results"
reports_dir = os.path.join(results_dir, "reports")
figures_dir = os.path.join(results_dir, "figures")
logs_dir = "logs"

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_file_path = os.path.join(logs_dir, "execution.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Execution started.")

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target
logging.info("Dataset loaded successfully.")

# Define models and their parameter grids
model_params = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "classifier__n_estimators": [10, 50, 100, 200]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {
            "classifier__C": [0.1, 1, 10, 100]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": [0.001, 0.01, 0.1, 1]
        }
    },
    "K-NN": {
        "model": KNeighborsClassifier(),
        "params": {
            "classifier__n_neighbors": [3, 5, 7, 9]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "classifier__max_depth": [3, 5, 7, 10]
        }
    }
}

# Dynamic number of splits for K-Fold
n_splits_values = [5, 10]
model_results = []

try:
    for n_splits in n_splits_values:
        logging.info(f"Evaluating models with n_splits={n_splits}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Perform K-Fold Cross-Validation
        for name, model_info in model_params.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model_info['model'])
            ])
            grid_search = GridSearchCV(pipeline, model_info['params'], cv=kf, n_jobs=-1, verbose=1, return_train_score=True)
            grid_search.fit(X, y)
            mean_val_score = np.mean(grid_search.cv_results_['mean_test_score'])
            std_val_score = np.std(grid_search.cv_results_['mean_test_score'])
            mean_train_score = np.mean(grid_search.cv_results_['mean_train_score'])

            logging.info(f"{name} (n_splits={n_splits}): Best Score = {grid_search.best_score_:.4f}, Mean Validation Score = {mean_val_score:.4f}, Std Dev = {std_val_score:.4f}")
            logging.info(f"Training Performance: Mean Train Score = {mean_train_score:.4f}")
            
            model_results.append({
                'name': name,
                'grid_search': grid_search,
                'n_splits': n_splits,
                'mean_val_score': mean_val_score,
                'std_val_score': std_val_score
            })

    # Bootstrap with .632 adjustment
    n_iterations = 100
    bootstrap_scores = {}
    scaler = StandardScaler()

    for name, model_info in model_params.items():
        logging.info(f"Running Bootstrap for model: {name}")
        model = model_info['model']
        if isinstance(model, LogisticRegression):
            model.set_params(max_iter=5000, tol=0.01, solver='saga')

        pipeline = make_pipeline(scaler, model)
        scores = []
        for i in range(n_iterations):
            if i % 10 == 0:
                logging.info(f"Iteration {i}/{n_iterations} for {name}")
            X_sample, y_sample = resample(X, y, n_samples=len(X))
            pipeline.fit(X_sample, y_sample)
            y_pred = pipeline.predict(X)
            err = 1 - accuracy_score(y, y_pred)
            loo_err = err
            err_632 = 0.368 * err + 0.632 * loo_err
            scores.append(1 - err_632)
        bootstrap_scores[name] = scores
        logging.info(f"Completed Bootstrap for model: {name}. Mean Score: {np.mean(scores):.4f}")

    # Write results to report
    report_path = os.path.join(reports_dir, "model_selection_report.txt")
    with open(report_path, "w") as report_file:
        report_file.write("Model Selection Report\n")
        report_file.write("======================\n\n")
        report_file.write("K-Fold Cross-Validation Results:\n")
        for result in model_results:
            report_file.write(f"Model: {result['name']} (n_splits={result['n_splits']})\n")
            report_file.write(f"  - Best Score: {result['grid_search'].best_score_:.4f}\n")
            report_file.write(f"  - Mean Validation Score: {result['mean_val_score']:.4f}\n")
            report_file.write(f"  - Std Dev: {result['std_val_score']:.4f}\n\n")
        
        report_file.write("Bootstrap Results:\n")
        for model_name, scores in bootstrap_scores.items():
            report_file.write(f"Model: {model_name}\n")
            report_file.write(f"  - Mean Score: {np.mean(scores):.4f}\n")
            report_file.write(f"  - Std Dev: {np.std(scores):.4f}\n\n")

    # Determine best models
    best_kfold_model = max(model_results, key=lambda x: x['grid_search'].best_score_)
    best_bootstrap_model = max(bootstrap_scores.items(), key=lambda x: np.mean(x[1]))

    logging.info(f"Best Model (K-Fold): {best_kfold_model['name']} with Score: {best_kfold_model['grid_search'].best_score_:.4f}")
    logging.info(f"Best Model (Bootstrap): {best_bootstrap_model[0]} with Score: {np.mean(best_bootstrap_model[1]):.4f}")

    # Visualizations
    fig1, axs1 = plt.subplots(1, 2, figsize=(16, 6))
    fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6))

    # K-Fold Visualization
    kfold_means = [result['mean_val_score'] for result in model_results if result['n_splits'] == 5]
    kfold_stds = [result['std_val_score'] for result in model_results if result['n_splits'] == 5]
    models = [result['name'] for result in model_results if result['n_splits'] == 5]
    sns.barplot(x=models, y=kfold_means, ax=axs1[0])
    axs1[0].set_title('Mean ± Std Dev of K-Fold Scores (n_splits=5)')
    axs1[0].set_xlabel('Model')
    axs1[0].set_ylabel('Score')
    axs1[0].errorbar(range(len(kfold_means)), kfold_means, yerr=kfold_stds, fmt='o', color='black')

    for n_splits in n_splits_values:
        scores = [result['mean_val_score'] for result in model_results if result['n_splits'] == n_splits]
        models = [result['name'] for result in model_results if result['n_splits'] == n_splits]
        axs1[1].plot(models, scores, marker='o', label=f'n_splits={n_splits}')
    axs1[1].set_title('K-Fold Scores Across Splits')
    axs1[1].legend()

    # Save K-Fold visualization
    fig1.savefig(os.path.join(figures_dir, "kfold_visualization.png"))

    # Bootstrap Visualizations
    for name, scores in bootstrap_scores.items():
        axs2[0].plot(range(n_iterations), scores, label=name)
    axs2[0].set_title('Bootstrap Accuracy Trends')
    axs2[0].legend()

    scatter_model = list(bootstrap_scores.keys())[0]
    axs2[1].scatter(range(len(bootstrap_scores[scatter_model])), bootstrap_scores[scatter_model], alpha=0.5)
    axs2[1].set_title(f'Scatter Plot of Bootstrap Scores ({scatter_model})')

    # Save Bootstrap visualization
    fig2.savefig(os.path.join(figures_dir, "bootstrap_visualization.png"))
    logging.info("All visualizations saved successfully.")
except Exception as e:
    logging.error(f"An error occurred: {e}")
    raise
