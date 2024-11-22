import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.LinearRegression import LinearRegressionModel
from Models.LinearRegression import Metrics

class Bootstrapping:
    def __init__(self, n_iter):
        self.n_iter = n_iter

    def bootstrap(self, X, y):

        n_rows = len(y)
        mse = []
        r2 = []

        for i in range(self.n_iter):
            bs_indis = np.random.choice(range( n_rows),   
                                               size=n_rows, 
                                               replace=True)
            oob_indis =  np.setdiff1d(np.arange( n_rows),   bs_indis)

            if len(oob_indis) == 0:
                continue

            X_train  = X[bs_indis] 
            X_val = X[oob_indis]
            y_train  = y[bs_indis] 
            y_val = y[oob_indis]
            
            
            reg_model = LinearRegressionModel()
            reg_model.fit(X_train, 
                      y_train)
            y_preds = reg_model.predict(X_val)

            mse_score = Metrics.mean_squared_error(y_val,y_preds)
            r2_score= Metrics.r_squared(y_val,y_preds)
            aic_score = Metrics.aic(y_val, y_preds, X_val)

            mse.append(mse_score)
            
            r2.append(r2_score)

            aic.append(aic_score)

            print(f"Iteration {i + 1}/{self.n_iter} - MSE: {mse_score:.4f}, R-Squared: {r2_score:.4f}, AIC: {aic_score:.4f}")

        return np.mean(  mse), np.mean(  r2), np.mean(aic)
