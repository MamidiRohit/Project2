import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..models.model_selector import ModelSelector 


def test_predict():
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    data = pd.read_csv(csv_path)
   

    X = data[['x_0', 'x_1', 'x_2']].values
    y = data['y'].values

# Initialize the model and ModelSelector
    model = LinearRegression()
    selector = ModelSelector(model, k=5, n_bootstraps=100)

# Evaluate the model using k-fold, bootstrap, and AIC
    results = selector.evaluate(X, y)

# Print the results
    print("k-fold CV MSE:", results["k_fold_cv"])
    print("Bootstrap MSE:", results["bootstrap"])
    print("AIC:", results["aic"])
    print("BIC:", results["bic"])

if __name__ == "__main__":
    test_predict()    
