import numpy as np
import pandas as pd

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

    def __init__(self,num_estimators,learning_rate=0.01,max_depth=5,min_split=2):
        '''
        Multiple regression trees are needed to gradually correct the errors of the previous tree through each tree.
        Ultimately, the weak classifier becomes stronger.
        '''
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_split = min_split
        self.num_estimators = num_estimators
        self.models = []

        
    def fit(self,X,y):
        y_pred = np.zeros(len(y))              # to store new prediction values
        for _ in range(self.num_estimators):
            tree = RegressionTree()
            # for mean squared error (MSE):
            # L(yi, f(xi)) = 1/2 * (yi - f(xi))^2
            # 
            # its partial derivative is:
            # ∂L/∂f(xi) = f(xi) - yi
            # 
            # after substituting into the gradient formula:
            # rim = yi - fm-1(xi)
            residual = y - y_pred
            tree.fit(X,residual)
            # when I use Mean Squared Error (MSE) to calculate, the output γ of each tree essentially represents the current residual value.
            # because, for MSE, the gradient is given by:
            # rim = yi - fm-1(xi)
            gamma = self.learning_rate * tree.predict(X)
            y_pred += gamma
            self.models.append(tree)


    def predict(self,X):

        y_pred = np.zeros(len(X))
        for model in self.models: # to sum the donation of every tree
            y_pred += self.learning_rate * model.predict(X)
        return y_pred

class RegressionTree():
    def __init__(self,max_depth=5,min_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_split
        self.tree = None

    def fit(self,X,y):
        def build_tree(X,y,depth):
            '''
            Since it is a regression tree, the function that builds the tree will call itself, 
            so it is written as a separate function instead of using the fit function.
            
            '''
            # if the current depth reaches the maximum depth or the number of samples is less than the minimum number of splits, return the mean number of leaf
            # we use MSE , therefore return mean(y) 
            if depth >= self.max_depth or len(y) < self.min_samples_split:
                return {"leaf_value": np.mean(y)}
            
            best_spilt_point = 1

            # there is no optimal split point, the tree does not need to split anymore
            if not best_spilt_point:
                
                return {"leaf_value": np.mean(y)}
            
            left_subtree = build_tree(X,y,depth)
            right_subtree = build_tree(X,y,depth)

            return {

                "left": left_subtree,
                "right": right_subtree
            }

        self.tree = build_tree(depth=0)

    def find_best_split_point(self,X,y):

        _, n_features = X.shape
        best_split = None
        best_error = float('inf')
        
        for index in range(n_features):
            values = X[:,index]
            # all possible segmentation points of the current feature
            dynamic_spilt = np.unique(values)

            for value in dynamic_spilt:

                left_indices = X[:index] <= value
                right_indices = X[:index] > value

                left_X, right_X = X[left_indices], X[right_indices]
                left_y, right_y = y[left_indices], y[right_indices]
                
                # Ensure both sides have at least one sample
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                current_error = self.regErr(left_y, right_y, mode=1)
                if current_error < best_error:
                    best_error = current_error
                    best_split = {
                        "feature_index": index,
                        "split_value": value,
                        "left_X": left_X,
                        "left_y": left_y,
                        "right_X": right_X,
                        "right_y": right_y
                    }

        return best_split                

    def predict(self,X):

        predictions = []

        for i in X:
            node = self.tree
            
            # loop through the tree until a leaf node is reached
            while "leaf_value" not in node:
                feature_index = node["feature_index"]
                split_value = node["split_value"]
                
                # according to the feature value and segmentation value of the current sample,
                # decide whether to move left or right
                if i[feature_index] <= split_value:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["leaf_value"])
        return np.array(predictions)

    def regErr(self, left_y, right_y, mode=1):
        if mode == 1:
            # calculate the mse
            left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
            right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
            return (left_mse + right_mse) / (len(left_y) + len(right_y))

