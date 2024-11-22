# CS584_Project2
K_OR_Boot_RThomson

Project 2 Readme

“K_OR_Boot”
Rebecca Thomson
A20548618
Nov. 2024

Filename: K_OR_Boot_RThomson.py
Summary:  This program automatically selects model attributes by using either K-Fold or Bootstrapping per the user’s request.  It uses linear regression.  It works on DataFrame datasets.

The call for this function is:
KFold_or_Boot(Xdata, Ydata, K_fold_or_boot="K-fold", K_fold_num=10, Boot_num_goal=50, model_type="linear", select_para=”Yes”)

Parameters:
Xdata,Ydata= Dataframes of the independent attribute variables and the dependent response variable respectively.  

K_fold_or_Boot= “Boot” an indicator if the application should run either a K-Fold or a Boot-strapping cross validation method to select the model.  Correct syntax is either “K-fold” or “Boot” 

K_fold_num=10  the number of K-folds requested.  The default is 10.  Individual datapoints within the dataset will be assigned randomly to each fold, resulting in the possibility of different results for each run.  The number of elements assigned to each fold shall vary slightly, due to the randomness.

Boot_num_goal=50  the number of sets of Bootstrapped datasets created.  The default is 50.  The size of each created dataset will match the original dataset’s size.

model_type="linear" A dummy argument meant to be used when additional model types were added.

Select_para= “Yes” To have the program automatically select a subset of parameters based upon sorting parameters by correlation.  The program starts with the most correlated attributes, and then adding one additional parameter for each model until all are used.  Will select model with lowest test error on the train/test split used in either K-Fold or Bootstrapping.  Selecting “No” will run either K-Fold or Bootstrapping only on the entire model provided with no parameter selection.

Results:
The K-fold option will return a list of the attributes/parameter names chosen by the program.  It also prints: 
-	A list of error terms= SSE/n for each model run.
-	The calculated average AIC for each model, using the K-fold training sets.
-	A list of the parameter names for the best model from the K-Fold CV.
The bootstrapping option will return:
-	A list of error terms = SSE/(Individual test set N*Number of Bootstrap sets) for each model run.
-	The calculated AIC for each model, using the entire data set, and each parameter sets.
-	A list of the parameter index values (starting from 1) for the best model from the Bootstrapping CV.
Single model (without parameter selection) will return both the calculated error and the AIC for the model for both K-Fold and Bootstrapping.

How this program calculates its results:
1.	The K-fold option of a model selector sticks closely to the “The Elements of Statistical Learning” Section 7.10.2 outline of the appropriate method for K-Fold cross-validation.
2.	If the program argument ‘select_para=”Yes”’, then the program will select the best set of most correlated parameters in the following method:
  a.	The Dataset is divided into K Folds of datapoints.
  b.	For each fold:
    i.	A correlation Matrix is created.
    ii.	Each independent variable’s correlation is compared to the dependent variable.
    iii.	The independent variables are ordered from most correlated to least correlated.  Absolute values are taken, because a correlation of exactly -1 is very correlated.
    iv.	The best number of parameters is determined by starting at the most correlated and adding additional parameters one by one to the set in iteration.  A regression is run on the data for each iteration, except the holdout fold K.  This produces a trained model for each number of parameters.
    v.	For each trained model, we use the test set- the withheld data from fold K- to predict the expected values of the response variable.
    vi.	We calculate the sum of squared error, and divide by N.
    vii.	Values are saved within a matrix.
  c.	After the error of all folds and all numbers of parameters are calculated, we sum up the error for all folds for each number of parameters.  The group of parameters that has the least error is selected.  This results in parameters selected that have the most correlation and the least error.
  d.	The program prints out:
    i.	 The SSE/N for each model in list format.
    ii.	The AIC for each model.
    iii.	The result of the model selection by printing out the given names of the attributes from the original dataframe in list format.  
    iv.	The names can also called by:  (class_instance).parameter_names
3.	If the ‘select_para=”No”’, the program does not attempt to select the best parameters, and will only perform K-Fold on the model as given.  The program will print out the SSE/N and AIC for this model.

4.	Bootstrapping: If parameter selection is chosen, the program orders and selects attributes as it did for the K-Fold method:

  a.	A correlation Matrix is created from the entire dataset.
  b.	Each independent variable’s correlation is compared to the dependent variable.
  c.	The independent variables are ordered from most correlated to least correlated.  Absolute values are taken, because a correlation of exactly -1 is very correlated.
  d.	Models to be tested are selected by the number of parameters is determined by starting at the most correlated and adding additional parameters one by one to the set in iteration.

  e.	For each model, several iterations of Bootstrapping are run, and the calculated test error is averaged for all runs of a model.  

    i.	N samples, which is the same number of samples of the original dataset, are randomly selected from the original dataset with replacement.  The selected datapoints are the training data for the   bootstrapping iteration.  
    ii.	The datapoints not selected are the test data for the individual bootstrapping iteration.
    iii.	Error for each iteration of each model type are saved and averaged as shown in Equation 7.54, p. 250 of “The Elements of Statistical Learning”.

5.	If Bootstrapping is chosen, but parameter selection should not be run, then the program will only run a Bootstrapping set on the full model.  The error and the AIC will be printed.
6.	AIC is calculated within the Linear Regression function, and uses the formula: N*log(RSS/N)+2*P with P being the number of parameters, and RSS calculated from a model trained on the entire dataset (with only the selected attributes)




Limitations and Possible failures:
1.	This program uses the correlation function provided by Python.  Because of this, categorical variables could have been dropped in the selection of variables- therefore they are forced into numeric format.
2.	This program tries to automatically drop any row that contains strings or NAN values.  If you leave a column of strings (labels, etc) every row will be dropped.  Please check your data carefully before use.
3.	This program does not have a set seed: for consistent return values, please set a seed.
4.	Currently, the AIC produced from the training sets in K-Fold when selecting various numbers of parameters.  I am trying to fix it.
5.	Currently, the parameters listed for best Bootstrapping are listed by original index, not by name.

What should be added:
1.	It was the intension that this program would take an argument that would select the type of model the user wanted to implement.  Such as Linear regression, Logistic regression, LDA, etc.  However, coding these models from first principals was beyond the time frame.
2.	The Bootstrapping is extremely computationally expensive, possibly altering the program to use more parallel processing with threads might be useful.
3.	Adding a plot function that would plot the SSR/N and AIC terms in comparison to parameter set size would make results more visually interpretable.  

K-Fold Cross-validation and bootstrapping model selection method both appear to agree with the AIC selector value derived from linear regression.

Libraries loaded by this program:
import numpy as np
import pandas as pd
#system, for exit if certain errors.
import sys
import random
#import matplotlib.pyplot as plt
import math

Here is an example of a possible program run, with arguments:

#Load and clean a dataset
df_auto1 = pd.read_csv('auto-mpg[1].csv')
df_auto = df_auto1.applymap(lambda x : pd.to_numeric(x,errors='coerce'))
df_auto=df_auto.iloc[:, :8]
df_auto=df_auto.dropna()
X = df_auto.iloc[:, 1:8]  # independent
y = df_auto.iloc[:, :1]  # dependent
#run K_fold
try22=KFold_or_Boot(X,y,K_fold_or_boot="K-Fold", K_fold_num=10,Boot_num_goal=50,model_type="linear",select_para="Yes")
 
#Results that are printed out:
Model parameter selection complete with K-Fold CV
The SSE/n for each model (# parameters sorted by correlation) is:
[11.5411, 11.5093, 11.476, 11.4943, 10.1574, 9.6734, 9.7167]
The AIC for each model is:
[452.4092, 451.1649, 449.8579, 450.5779, 398.1693, 379.1961, 380.8966]
List of parameter names for best model from K-fold CV:
['weight', 'displacement', 'horsepower', 'cylinders', 'origin', 'model year']
 


#Run Bootstrapping model selection:
try22=KFold_or_Boot(X,y,K_fold_or_boot="Boot", K_fold_num=10,Boot_num_goal=50,model_type="linear",select_para="Yes")
 

#Expected Results:
Model parameter selection complete with Bootstrapping
The average error for each model (# parameters sorted by correlation) is:
[18.8646, 18.8272, 18.2903, 18.4348, 11.9554, 11.3466, 11.6954]
The AIC for each model is:
[502.3489, 500.8027, 498.2238, 499.8186, 429.4318, 420.1433, 421.8489]
Index of parameters for best model from Bootstrapping:
[4, 2, 3, 1, 6, 7]
