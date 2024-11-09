import numpy as np


class GBT():
    '''
    parameters:
    for function fit, get the inputed training data X and target variable y
    init fuction needs learning rate η, and the maximum depth d of the decision tree model
    
    initialize the model : let F0(x) = 0
    then, need a loop , in the loop performing:
        calculate the residuals of the loss function rmi: rmi = yi - Fm-1(xi), where xi and yi are the features and target value of the i-th sample, respectively
        train a regression tree Gm(X) and predict the residual value for each xi
        minimize the mean square error of each node to find regression coefficients of leaf nodes
        update the model: Fm(x) = Fm-1(x) + ηgammamGm(x)
    return final model
    '''

    def __init__(self,loss,learning_rate=0.01,max_depth=5,min_split=2):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_split = min_split
        self.loss = loss
    def fit(self,X,y):
        y_pred = np.zeros(len(y))#用于存储新的预测值
        for i in range():
            tree = RgressionTree()

        pass
    def predict():
        pass

class RgressionTree():
    def __init__(self,max_depth=5,min_split=2):
        pass
    def fit(self,X,y):
        pass
    def predict(self,X):
        pass
    def regErr():
        pass
    def createTree():
        pass
    def split():
        pass
