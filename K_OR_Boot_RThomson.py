#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 02:37:08 2024

@author: zekkers
"""

#libraries
import numpy as np
import pandas as pd
#system, for exit if certain errors.
import sys
import random
#import matplotlib.pyplot as plt
import math
#Note: this is a warning that is not a full error.  I removed it because this 
#passes testing.
pd.options.mode.chained_assignment = None  # default='warn'

class KFold_or_Boot:
    
    def check_data(self,fullxs,fullys):
        #checking that input is dataframe.
        
        if isinstance(fullxs,list):
            fullxs=pd.DataFrame(fullxs)
        if isinstance(fullys,list):
            fullys=pd.DataFrame(fullys)
            
        #Checking length of X and Y are equal.
        if (len(fullys) != len(fullxs)):
            print("Dependant and response variables not equal length.  check data.  Analysis stopped")
            tempstatus=1
            sys.exit(tempstatus)
        Combined=pd.concat([fullys, fullxs], axis=1, ignore_index=True)   
        
        df_auto = Combined.applymap(lambda x : pd.to_numeric(x,errors='coerce'))
        df_auto=df_auto.iloc[:, :]
        df_auto=df_auto.dropna()
        #df_auto
        #print(df_auto)
        X1 = df_auto.iloc[:, 1:]  # independent

        X=X1.applymap(lambda x : float(x))
        #X=X.dropna()
        #print(X)
        y = df_auto.iloc[:, :1]  # dependent
        
        return X,y
    
    def __init__(self, Xdata,Ydata,K_fold_or_boot="K-fold", K_fold_num=10,Boot_num_goal=50,model_type="linear",select_para="Yes"):
        
        X,y=self.check_data(Xdata,Ydata)
        
        self.Xdata=X
        self.Ydata=y
        #The total number of Parameters is P
        self.P_num= int(len(Xdata.columns))
        if (select_para=="Yes" or select_para=="YES" or select_para=="Y" or select_para=="y"):
            self.P_num_start=0
        else:
            self.P_num_start=self.P_num-1
        self.Xdata = self.Xdata.dropna(subset=self.Xdata.columns.values)
        self.Ydata = self.Ydata.dropna(subset=self.Ydata.columns.values)
        self.N=float(len(Xdata))
        self.master_batch=[]
        if ( K_fold_or_boot=="Boot"):
            #only Boot
            self.Boot_perform(Boot_num_goal, model_type= "linear")
        elif (K_fold_or_boot=="K-fold"):   
            self.K_fold_perform(Xdata, Ydata, K_fold_num, model_type= "linear")
            #only K-fold
            
        else:
            #incorrect call, this is an error.
            print("Neither K-Fold cross-validation nor Bootstrapping choosen.  Please check function call.")
        
                
    def order_by_correlation_per_fold(self,tempXtrain,tempYtrain,tempXtest):
        #Function to determine most correlated attributes per fold, and return the X data in order
        #Make one dataframe of y,X of this fold's training data
        self.z=pd.concat([tempYtrain, tempXtrain], axis=1, ignore_index=True)
        #Extract the first row of matrix, excluding corr(Y,Y) in ABSOLUTE VALUE
        
        self.temp_corr=self.z.corr().iloc[:1,1:].abs()
        #Creating a list with each attribute and their correlation value to Y
        #Correlation function creates Series data, it is hard to sort
        templist=self.temp_corr.values.tolist()
        tempnames=list(self.temp_corr)
        h=[]
        #nam=[]
        for i in range(0,len(templist[0])):
            t=[]
            t.append(tempnames[i])
            t.append(templist[0][i])
            h.append(t)
        #using lists to create dataframe, sort by correlation
        test2=pd.DataFrame(h, columns=['attributes','correlation'])
        df=test2.transpose()
        df=df.sort_values(by='correlation', ascending=False, axis=1)
        #print(df) #to determine if all parameters are making it through corr()
        #re-organize X data train and test for ease of itterating regressions later.
        temploc=int(df.iloc[0,0])-1
        tempX1=pd.DataFrame(tempXtrain.iloc[:,temploc])
        tempX2=pd.DataFrame(tempXtest.iloc[:,temploc])

        for i in range(1,len(h)):
            tempX3=pd.DataFrame(tempXtrain.iloc[:,int(df.iloc[0,i])-1])
            tempX4=pd.DataFrame(tempXtest.iloc[:,int(df.iloc[0,i])-1])
            tempX1=tempX1.join(tempX3)
            tempX2=tempX2.join(tempX4)
        #plt.plot(tempX1.columns,df[1:][1])
        return(tempX1,tempX2,df)
    
    def K_fold_perform(self,Xdata, Ydata, K_fold_num, model_type= "linear"):
        #function to perform K_fold for all folds and all numbers of parameters.
        #Returns 
        
        #Create an index list of length(Xdata) that has K_Fold_num randomly assigned classes.  These will be the groups withheld each time.
        K_index= []
        for i in list(range(len(Xdata))):
            K_index.append(random.randint(1,K_fold_num))
        #Create list to save error terms of each fold (K columns) and each # of parameters used (P rows)
        #Also create AIC list to compare the models
        self.err_terms = []
        self.AIC_terms = []
        corr_terms_for_all_folds=[]
        for i in list(range(K_fold_num+1)):
            lst= list(range(self.P_num))
            self.err_terms.append(lst)
            self.AIC_terms.append(lst)
        #create a list of parameters univariate correlation with the ydata labels. There will be K groups of data to correlate, then sort. 
        #First, create K groups of data not withheld and withheld for future testing. Create masterlist of all batches of data.
        
        #for each fold
        for k in list(range(1,K_fold_num+1)):
            tempXtrain=pd.DataFrame()
            tempXtest=pd.DataFrame()
            tempYtrain=pd.DataFrame()
            tempYtest=pd.DataFrame()
            #for length of data
            for i in range(len(Xdata)):
                #if not in this fold
                if (K_index[i]!=k):
                    #print(Xdata.iloc[i])
                    tempXtrain=tempXtrain.append(Xdata.iloc[i],ignore_index=True)
                    #print(tempXtrain)
                    tempYtrain=tempYtrain.append(Ydata.iloc[i],ignore_index=True)
                else:
                    tempXtest=tempXtest.append(Xdata.iloc[i],ignore_index=True)
                    tempYtest=tempYtest.append(Ydata.iloc[i],ignore_index=True)

            #create re-sorted list of attributes with most correlation per this fold
            tempXtrain_sorted,tempXtest_sorted,corr_order=self.order_by_correlation_per_fold(tempXtrain,tempYtrain,tempXtest)
            corr_terms_for_all_folds.append(corr_order)
            
            #save sorted Xtrain for later model selection.
            
            self.master_batch.append(list(tempXtrain_sorted.columns))
            #Itterate over 1-Pnum, selecting the parameters with the most corralation each time.
            #Add the error for each itteration to the error array as the (i)fold, with (p) parameters.
            if (self.P_num_start==0):
                for p in list(range(0,self.P_num)):
                
                    self.err_terms[k][p],self.AIC_terms[k][p]=self.Linear_regression_error(tempXtrain_sorted.iloc[:,:p+1],tempXtest_sorted.iloc[:,:p+1],tempYtrain,tempYtest)
                
                
                    #save the term [#parameters, K_fold_num]
            else:
                self.err_terms[k][self.P_num_start],self.AIC_terms[k][self.P_num_start]=self.Linear_regression_error(tempXtrain_sorted,tempXtest_sorted,tempYtrain,tempYtest)
                
        #find the sum of all columns of # parameters.  Divide by N
        
        sum_fold_errors=[ round(sum(x)/self.N,4) for x in zip(*self.err_terms[1:]) ]
        avg_AIC_per_model=[ round(sum(x)/K_fold_num,4) for x in zip(*self.AIC_terms[1:]) ]
        
        #For model selection:
        if (self.P_num_start==0):
            print("Model parameter selection complete with K-Fold CV")
            print("The SSE/n for each model (# parameters sorted by correlation) is:")
            print(sum_fold_errors)
            print("The AIC for each model is:")
            print(avg_AIC_per_model)
            #When all folds are done:       
            #average error term for each fold. Lowest error term tells how many parameters/attributes to use in model selection.
            #Which model is the best?  Return index of min value of sum_fold_errors
            minpos = sum_fold_errors.index(min(sum_fold_errors))
            #print(minpos)
            parameter_names_index=[]
            for names in list(range(0,minpos+1)):
                parameter_names_index.append(int(corr_order.iloc[0,names])-1)
            #print(parameter_names_index)
            self.parameter_names=[]
            for names in parameter_names_index:
                self.parameter_names.append(Xdata.columns[names])
            print("List of parameter names for best model from K-fold CV:")
            print(self.parameter_names)
        else:
            #only print out index at P_num
            print("K-Fold on entire given model has been performed")
            print("The SSE/n for this model is:")     
            print(round(sum_fold_errors[self.P_num_start],4))
            print("The AIC for this model is:")
            print(round(avg_AIC_per_model[self.P_num_start],4))
        sum_fold_errors=[]
    
        
    def Boot_perform(self, Boot_num_goal, model_type= "linear"):
        #create list for all returned errors of bootstrapped sets
        Boot_set_errors=[0]*Boot_num_goal
        #Boot_set_AIC=[0]*Boot_num_goal
        #Boot_set_errors[1]=self.each_boot(self.Xdata,self.Ydata)
        if (self.P_num_start==0):
            
            #Running multiple iterations of Bootstrapping with various predictors.
            #First, sort the X data parameters by correlation.  We will use the previous function
            tempX_sorted,garbage,corr_order=self.order_by_correlation_per_fold(self.Xdata,self.Ydata,self.Xdata)
            #for all P sets, collect AIC and err_terms
            self.AIC_terms = [0]*self.P_num
            self.err_terms = [0]*self.P_num
            #for all P sets, run a Bootstrapping
            for p in list(range(0,self.P_num)):
                Boot_set_errors=[0]*Boot_num_goal
                #run LR for AIC term for this P set.  don't worry about error- it's garbage anyway.
                garbage2,self.AIC_terms[p]=self.Linear_regression_error(tempX_sorted.iloc[:,:p+1],tempX_sorted.iloc[:,:p+1],self.Ydata,self.Ydata)
                for i in range(0, Boot_num_goal):
                    #run individual Bootstraps for error for each itt
                    Boot_set_errors[i],garbage=self.each_boot(tempX_sorted.iloc[:,:p+1],self.Ydata)
                #average results for batch
                self.err_terms[p]=round((sum(Boot_set_errors))/(Boot_num_goal),4)
            #Now let's select best models.
            minpos = self.err_terms.index(min(self.err_terms))
            #print(minpos)
            parameter_names_index=[]
            for names in list(range(0,minpos+1)):
                parameter_names_index.append(int(corr_order.iloc[0,names])-1)
            #print(parameter_names_index)
            self.parameter_names=[]
            for names in parameter_names_index:
                self.parameter_names.append(self.Xdata.columns[names])
            print("Model parameter selection complete with Bootstrapping")
            print("The average error for each model (# parameters sorted by correlation) is:")
            print(self.err_terms)
            print("The AIC for each model is:")
            print(self.AIC_terms)
            print("Index of parameters for best model from Bootstrapping:")
            print(self.parameter_names)
                                                                      
        else:
            #Running only the full model 
            #getting AIC
            garbage4,set_AIC=self.Linear_regression_error(self.Xdata,self.Xdata,self.Ydata,self.Ydata)
            for i in range(0, Boot_num_goal):
                Boot_set_errors[i],garbage3=self.each_boot(self.Xdata,self.Ydata)
            #print(Boot_set_errors)
            avg_set_error=round((sum(Boot_set_errors))/(Boot_num_goal),4)
            #avg_set_AIC=round((sum(Boot_set_AIC))/(Boot_num_goal),4)
            print ("The average error for all bootstrapped sets of this model is:")
            print(avg_set_error)
            print ("The average AIC for all bootstrapped sets of this model is:")
            print(set_AIC)
        
    def each_boot(self,X,y):
        #randomly assign len(X) samples to bootstrap set.
        #create empty test/train datasets
        X_train = pd.DataFrame(columns=X.columns) # independent
        y_train = pd.DataFrame(columns=y.columns)  # dependent
        X_test = pd.DataFrame(columns=X.columns)  # independent
        y_test = pd.DataFrame(columns=y.columns)  # dependent 
        
        #list of rows that will be used
        used_samples=[0]*(len(X)+1)
        for i in range(0, len(X)):
            temp_index=random.randint(0,len(X)-1) #Randomly picking a index number within the index range
            
            used_samples[temp_index]=1 #marking that sample as used
             #put in temp training set
            X_train=pd.concat([X_train, X.iloc[[temp_index]]], axis=0, ignore_index=True)
            y_train=y_train.append(y.iloc[temp_index],ignore_index=True) #put in temp training set
        #taking unused samples into temp testing set.
        for i in range(0, len(X)):
            if(used_samples[i]==0): #check each index
                X_test=X_test.append(X.iloc[i],ignore_index=True)  #Insert into test set 
                y_test=y_test.append(y.iloc[i],ignore_index=True)  #Insert into test set         
        #print(used_samples)
        
        temp_err,temp_AIC=self.Linear_regression_error(X_train,X_test,y_train,y_test)
        temp_error=temp_err/(len(X_test))
        return temp_error,temp_AIC
        
    def Linear_regression_error(self,xtrain,xtest,ytrain,ytest):
        xtrain["intercept"] = 1
        xtest["intercept"] = 1
        X_train_matrix=xtrain.to_numpy()
        X_test_matrix=xtest.to_numpy()
        y_train_matrix=ytrain.to_numpy()
        y_test_matrix=ytest.to_numpy()
        
        
        
        B = np.linalg.inv(X_train_matrix.T @ X_train_matrix) @ X_train_matrix.T @ y_train_matrix
        #B.index = X.columns
        predictions_test = X_test_matrix @ B
        predictions_ytrain = X_train_matrix @ B
        self.error_train=0
        self.error_test=0
        para_num=int(len(xtrain.columns))
        N=len(xtrain)
        for s in list(range(len(ytest))):
            self.error_test+=((predictions_test[s]-y_test_matrix[s])*(predictions_test[s]-y_test_matrix[s]))
        for s in list(range(len(ytrain))):
            self.error_train+=((predictions_ytrain[s]-y_train_matrix[s])*(predictions_ytrain[s]-y_train_matrix[s]))
        
        
        AIC_temp=round(N*math.log10(self.error_train/N)+2*para_num,4)  #N*log(RSS/N) + 2k
        return(self.error_test.item(),AIC_temp)
         



'''
# For testing only

df_auto1 = pd.read_csv('auto-mpg[1].csv')
df_auto = df_auto1.applymap(lambda x : pd.to_numeric(x,errors='coerce'))
df_auto=df_auto.iloc[:, :8]
df_auto=df_auto.dropna()
#df_auto
#print(df_auto)
X1 = df_auto.iloc[:, 1:8]  # independent

X=X1.applymap(lambda x : float(x))
#X=X.dropna()
#print(X)
y = df_auto.iloc[:, :1]  # dependent





try22=KFold_or_Boot(X,y,K_fold_or_boot="Boot", K_fold_num=10,Boot_num_goal=50,model_type="linear",select_para="Y")
'''  