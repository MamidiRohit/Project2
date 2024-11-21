import numpy as np

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.prediction = None
    
    def find_split(self, X, y):
        best_score = float('inf')
        best_feature = 0
        best_value = 0
        
        # Try features
        for feature in range(X.shape[1]):
            # Try unique values for split
            values = np.unique(X[:, feature])
            
            for val in values:
                left_idx = X[:, feature] <= val
                right_idx = ~left_idx
                
                # Skip if split empty
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                
                # Calculate mean squared error
                left_pred = np.mean(y[left_idx])
                right_pred = np.mean(y[right_idx])
                
                left_score = np.sum((y[left_idx] - left_pred)**2)
                right_score = np.sum((y[right_idx] - right_pred)**2)
                score = left_score + right_score
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_value = val
        
        return best_feature, best_value
    
    def fit(self, X, y, depth=0):
        # Make a leaf node if at max depth
        if depth >= self.max_depth:
            self.prediction = np.mean(y)
            return
        
        # Find best split
        self.split_feature, self.split_value = self.find_split(X, y)
        
        # Split data
        left_idx = X[:, self.split_feature] <= self.split_value
        right_idx = ~left_idx
        
        # Create child nodes
        if left_idx.sum() > 0:
            self.left = DecisionTree(self.max_depth)
            self.left.fit(X[left_idx], y[left_idx], depth + 1)
        if right_idx.sum() > 0:
            self.right = DecisionTree(self.max_depth)
            self.right.fit(X[right_idx], y[right_idx], depth + 1)
            
        # If cant split, make leaf
        if self.left is None and self.right is None:
            self.prediction = np.mean(y)
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            node = self
            # Go until hit a leaf
            while node.prediction is None:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.prediction
            
        return predictions

class GradientBoosting:
    def __init__(self, n_trees=100, lr=0.1, max_depth=3):
        self.n_trees = n_trees
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None
    
    def fit(self, X, y):
        # Start with mean prediction
        self.base_pred = np.mean(y)
        current_pred = np.full_like(y, self.base_pred)
        
        # Add trees
        for _ in range(self.n_trees):
            # Fit tree to residuals
            residuals = y - current_pred
            tree = DecisionTree(self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            current_pred += self.lr * tree.predict(X)
            self.trees.append(tree)
    
    def predict(self, X):
        pred = np.full(len(X), self.base_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred