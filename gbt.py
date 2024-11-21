import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error

class GBT():
    '''
    parameters:
    for function fit, get the inputed training data X and target variable y
    init fuction needs learning rate η, and the maximum depth d of the decision tree model
    
    initialize the model : let F0(x) = 0
        calculate the residuals of the loss function rmi: rmi = yi - Fm-1(xi), where xi and yi are the features and target value of the i-th sample, respectively
        train a regression tree Gm(X) and predict the residual value for each xi
        minimize the mean square error of each node to find regression coefficients of leaf nodes
        update the model: Fm(x) = Fm-1(x) + ηgammamGm(x)
    return final model
    '''

    def __init__(self,num_estimators,max_depth=5,min_split=2,learning_rate=0.01,criterion = 'mse'):
        '''
        Multiple regression trees are needed to gradually correct the errors of the previous tree through each tree.
        Ultimately, the weak classifier becomes stronger.
        '''
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_split = min_split
        self.num_estimators = num_estimators
        self.criterion = criterion
        self.models = []

        
    def fit(self,X,y):
        # convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        y_pred = np.zeros(len(y))              # to store new prediction values
        if self.criterion != 'mas':
            self.criterion = 'mse'
        for _ in range(self.num_estimators):

            
            tree = RegressionTree(max_depth=self.max_depth,min_split=self.min_split, criterion = self.criterion) # needs to write the parameters
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
    def __init__(self,max_depth=5,min_split=2, criterion = 'mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_split
        self.criterion = criterion
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
            
            best_spilt_point = self.find_best_split_point(X,y)

            # there is no optimal split point, the tree does not need to split anymore
            if not best_spilt_point:
                
                return {"leaf_value": np.mean(y)}

            left_subtree = build_tree(best_spilt_point["left_X"], best_spilt_point["left_y"], depth + 1)
            right_subtree = build_tree(best_spilt_point["right_X"], best_spilt_point["right_y"], depth + 1)


            return {
                "feature_index":best_spilt_point["feature_index"],
                "split_value":best_spilt_point["split_value"],
                "left": left_subtree,
                "right": right_subtree
            }

        self.tree = build_tree(X,y,depth=0)

    def find_best_split_point(self,X,y):

        _, n_features = X.shape
        best_split = None
        best_error = float('inf')
        
        for index in range(n_features):
            values = X[:,index]
            # all possible segmentation points of the current feature
            dynamic_spilt = np.unique(values)

            for value in dynamic_spilt:

                left_indices = X[:,index] <= value
                right_indices = X[:,index] > value


                left_X, right_X = X[left_indices], X[right_indices]
                left_y, right_y = y[left_indices], y[right_indices]
                # ensure both sides have at least one sample
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

        if mode == 'mae':
            left_mae = np.mean(np.abs(left_y - np.mean(left_y))) * len(left_y) if len(left_y) > 0 else 0
            right_mae = np.mean(np.abs(right_y - np.mean(right_y))) * len(right_y) if len(right_y) > 0 else 0
            return (left_mae + right_mae) / (len(left_y) + len(right_y))
        else:
            # calculate the mse, cause it is default
            left_mse = np.var(left_y) * len(left_y) if len(left_y) > 0 else 0
            right_mse = np.var(right_y) * len(right_y) if len(right_y) > 0 else 0
            return (left_mse + right_mse) / (len(left_y) + len(right_y))



def train_and_save_model():
    # load the Iris dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the GBT model
    model = GBT(num_estimators=20, max_depth=3, min_split=10, learning_rate=0.1, criterion='mse')
    model.fit(X_train, y_train)

    # predict and calculate accuracy
    y_pred = model.predict(X_test)
    y_pred_class = np.round(y_pred).astype(int)
    y_pred_class = np.clip(y_pred_class, 0, 2)
    accuracy = accuracy_score(y_test, y_pred_class)

    print("Predicted values of y (rounded to nearest class):", y_pred_class)
    print("True values of y:", y_test)
    print("Classification Accuracy:", accuracy)

    # save model
    with open('gbt_iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

    # plot the predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value (Class Labels)')
    plt.title('GBT Predictions vs True Values on Iris Dataset')
    plt.legend()
    plt.savefig('GBT_Predictions_Iris.png')
    plt.show()


def load_and_plot_model():
    # load the model from file
    if not os.path.exists('./gbt_iris_20_3_10_01.pkl'):
        print("No saved model found. Please22 train the model first.")
        return

    with open('./gbt_iris_20_3_10_01.pkl', 'rb') as f:
        model = pickle.load(f)

    data = load_iris()
    X, y = data.data, data.target

    # split the data into training and testing sets (test set is 1/3 of the total data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

    # predict
    y_pred = model.predict(X_test)
    y_pred_class = np.round(y_pred).astype(int)
    y_pred_class = np.clip(y_pred_class, 0, 2)

    # calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    print("Classification Accuracy on the test dataset:", accuracy)

    # plot the predictions vs true values for the test set
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Sample Index (Test Set)')
    plt.ylabel('Target Value (Class Labels)')
    plt.title('GBT Predictions vs True Values on Test Set (Loaded Model)')
    plt.legend()
    plt.show()


def train_concrete_model(file_path='./Concrete_Data.xls'):
    # load the dataset
    if not os.path.exists(file_path):
        print(f"{file_path} file not found. Please ensure it is in the correct directory.")
        return
    
    dataset_name = os.path.basename(file_path).split('.')[0]  # extract the base name without extension
    print(f"Loading dataset: {file_path}")

    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1].values  # all columns except the last one are features
    y = data.iloc[:, -1].values   # the last column is the target variable
    
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the GBT model
    model = GBT(num_estimators=40, max_depth=5, min_split=10, learning_rate=0.08, criterion='mse')
    model.fit(X_train, y_train)

    # predict and calculate RMSE and R² Score
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)  # R² Score

    # print evaluation metrics
    print(f"Evaluation Metrics for Dataset '{dataset_name}':")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # save the model with the appropriate name
    model_file_name = f'gbt_{dataset_name}_model.pkl'
    with open(model_file_name, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully as '{model_file_name}'")

    # plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values', alpha=0.6, marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value (Compressive Strength)')
    plt.title(f'GBT Predictions vs True Values on Dataset: {dataset_name}')
    plt.legend()
    plot_file_name = f'GBT_Predictions_{dataset_name}.png'
    plt.savefig(plot_file_name)
    plt.show()
    print(f"Prediction plot saved as '{plot_file_name}'")


if __name__ == "__main__":
    while True:
        print("\nSelect an option:")
        print("1. Train Iris dataset, test it on test set and save model")
        print("2. Load saved Iris model and plot predictions")
        print("3. Train default Concrete_Data.xls dataset, test it, and save model")
        print("4. Train custom dataset (input file name), test it, and save model")
        print("q. Exit")

        choice = input("Enter your choice (1/2/3/4/q): ")

        if choice == '1':
            train_and_save_model()
        elif choice == '2':
            load_and_plot_model()
        elif choice == '3':
            train_concrete_model()  # default Concrete_Data.xls
        elif choice == '4':
            print("Please modify the GBT parameters in the code if needed before proceeding!")
            file_name = input("Enter the dataset file name (including extension, e.g., 'your_data.xls'): ")
            train_concrete_model(file_path=file_name)  # custom dataset
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or q.")
            print("Using default choice 1")
            train_and_save_model()




