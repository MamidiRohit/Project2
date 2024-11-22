import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_regressiontree_results(y, pred):
    residuals = pred - y
    plt.figure(figsize=(10, 6))
    plt.scatter(y, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
    rmse = np.sqrt(np.mean(residuals ** 2))
    plt.xlabel('True Values')
    plt.ylabel('Residuals (Predicted - True)')
    plt.title(f'Residual Plot\nRMSE: {rmse:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_classificationtree_results(y,preds):
    cm = confusion_matrix(y, preds)
    classes = sorted(list(set(np.concatenate([y, preds]))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes,yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label\nTotal Predictions: {len(preds)}')
    accuracy = np.sum(np.array(preds) == np.array(y)) / len(y) * 100
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.1f}%')
    plt.tight_layout()
    plt.show()
