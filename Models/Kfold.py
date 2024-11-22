import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.LinearRegression import LinearRegressionModel
from Models.LinearRegression import Metrics

class CrossVal: 
    def __init__(self, k):
        self.k = k
    
    def kFold(self, X, y):
        indis = np.arange(len(y))
        np.random.seed(10)
        np.random.shuffle(indis)
        foldSize = int(len(y)/ self.k)
        mse = []
        r2 = []
    
        for i in range(self.k):
            val_indis = indis[i*foldSize:(i+1) * foldSize]
            train_indis = np.setdiff1d(indis, val_indis)
            X_train  = X[train_indis] 
            X_val = X[val_indis]
            y_train =  y[train_indis] 
            y_val =  y[val_indis]
            
            reg_model = LinearRegressionModel()
            reg_model.fit(X_train, 
                      y_train)
            y_preds = reg_model.predict(X_val)
            
            mse.append(Metrics.mean_squared_error(y_val, 
                                                  y_preds))
            r2.append(Metrics.r_squared(y_val, 
                                        y_preds))
            
        return np.mean(mse), np.mean(r2)
