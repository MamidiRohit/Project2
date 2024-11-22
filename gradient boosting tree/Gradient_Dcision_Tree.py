import numpy as np

# tree node class
class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature=feature
        self.thre = threshold
        self.left = left
        self.right = right

        self.val = value
    
    def is_leaf(self):
        return self.val is not None

# decision tree regressor class
class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth


    # training data
    def fit(self, X, Y):
        # if X and Y are dataframe, change to np.ndarray
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            X = X.to_numpy()
            Y = Y.to_numpy()

        # build decision tree
        self.root = self.build_tree(X, Y, curr_depth=0)
    

    # predict 
    def predict(self, X):
        return np.array([self.predict_sample(sample, self.root) for sample in X])


    # build a decision tree based on the train data
    def build_tree(self, X, Y, curr_depth=0):
        # stop training
        if curr_depth >= self.max_depth or len(Y) < self.min_samples_split or np.unique(Y).size == 1:
            return Node(value=np.mean(Y))

        # find best split
        bestSplit = self.best_split(X, Y)
        # No split to imrpove MSE
        if bestSplit['mse'] == float("inf"):
            return Node(value=np.mean(Y))
        

        # continue to find other subtrees
        left_sub = self.build_tree(bestSplit['X_left'], bestSplit['Y_left'], curr_depth + 1)
        right_sub = self.build_tree(bestSplit["X_right"], bestSplit["Y_right"], curr_depth + 1)
        return Node(feature=bestSplit["feature"], threshold=bestSplit["thre"], left=left_sub, right=right_sub)


    # find best split
    def best_split(self, X, Y):
        # n is number of sample, and m is number of features
        n, m = X.shape
        bestSplit = {"mse": float("inf")}

        # traverse each feature
        for i in range(m):
            # potential thresholds in the feature
            thres= np.unique(X[:, i])
            # one of potential threshold in potential thresholds
            for j in thres:
                # split the data
                left_index = X[:, i] <= j
                right_index = X[:, i] > j

                # pass this threshold
                if len(Y[left_index]) == 0 or len(Y[right_index]) == 0:
                    continue

                # calculate weighted MSE for determining the best split
                mse = self.calculate_MSE(Y[left_index], Y[right_index])
                # get the threshold
                if mse < bestSplit["mse"]:
                    bestSplit['mse'] = mse
                    bestSplit['feature'] = i
                    bestSplit['thre'] = j
                    bestSplit['X_left'] = X[left_index]
                    bestSplit['X_right'] = X[right_index]
                    bestSplit['Y_left'] = Y[left_index]
                    bestSplit['Y_right'] = Y[right_index]

        return bestSplit


    # calcualte weighted MSE
    def calculate_MSE(self, left, right):
        # get left MSE and right MSE
        mse_left = np.mean((left - np.mean(left)) ** 2) if len(left) > 0 else 0
        mse_right = np.mean((right - np.mean(right)) ** 2) if len(right) > 0 else 0

        tol_samples = len(left) + len(right)
        weighted_mse = (len(left) / tol_samples) * mse_left + (len(right) / tol_samples) * mse_right
        
        return weighted_mse

    # predict value for the sample
    def predict_sample(self, sample, node):
        if node.is_leaf():
            # print(node.val)
            return node.val
        
        # continue to traverse the tree
        if sample[node.feature] <= node.thre:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)



class GradientBoostingTree():
    def __init__(self, n_estimators=100, rate=0.1, max_depth=3, min_samples_split=2, tol=10):
        # n estimater, learning rate, max depth of the tree, min samples split, tolerance
        self.rate = rate
        self.n_estimators = n_estimators
        self.depth = max_depth
        self.min_samples_split = min_samples_split
        # int value to record the number of no improvement 
        self.tol = tol

        # store each function
        self.trees = []

    def fit(self, X, Y):
        # inital prediction and residual
        self.init_pred = np.mean(Y)
        resi = Y - self.init_pred
        ### ealy stopping
        self.best_lost = float('inf')

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.depth, min_samples_split=self.min_samples_split)

            # fit X and resi from the last function
            tree.fit(X, resi)
            
            # predict residual
            pred = tree.predict(X)

            ### early stop 
            # use MSR to calculate the loss
            loss = np.mean((resi - pred) ** 2)
            # check whether improve
            # if no, record it
            if loss < self.best_lost:
                self.best_lost = loss
                # reset the patient to ensure no improvement is sequential
                self.tol = 10
            else:
                self.tol -= 1

            # if no improve is more than 10, break
            if self.tol == 0:
                break

            resi -= self.rate * pred

            self.trees.append(tree)

    # predict the data 
    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred)

        for i in self.trees:
            pred += self.rate * i.predict(X)

        return pred