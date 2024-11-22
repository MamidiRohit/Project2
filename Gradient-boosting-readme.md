# Gradient Boosting Tree Model Documentation

# Team Members:
A20546614 –PRAVEEN KUMAR GUDE
A20546967 – ABHARNAH RAJARAM MOHAN
A20529571–KARUTURI VEERENDRA GOPICHAND
A20561552 – BEZAWADA SAI SRAVANYA

# Overview:
This project is about creating a Gradient Boosted Trees (GBT) model from the ground up using Python. It sets up a solid way to learn with supervision, mainly for tasks that involve predicting numbers. Made for Google Colab, this project shows how easy it is to build machine learning models without using ready-made libraries like Scikit-learn or XGBoost. Users can adjust settings to fit their specific data and issues.The main goal here is to help people really understand how GBT works, both through theory and hands-on examples. The project is designed to be user-friendly, easy to test, and simple to expand.

# Key components of project:
1.Main Code (`gbt.py`):
    This file runs the Gradient Boosted Trees (GBT) method using Python without any extra libraries like Scikit-learn. It has:
    - `__init__` Method: Sets up the model with options you can change:
    - `num_est`: Number of trees in the model.
    - `learn_quan`: Learning rate that affects how much each tree adds to the model.
    - `high_dep`: The deepest the decision trees can go.
    - `vis_process`: An option to see the training steps.
    - Tree Building:
        - The `grow` method breaks down the data to build decision trees and reduce errors.
            - It keeps track of which features are most important while making the trees.
    -Making Predictions:
        - `_predict_`: Private method that goes through the trees to give predictions for single data points.
        - `predict`: Public method that predicts results for entire datasets using all the trees.
    - Training the Model:
        - `fit`: Trains the model by adding trees one by one, each one correcting mistakes from the last.
            - It checks the Mean Squared Error (MSE) to see how well it’s doing at each step.
    - Analysis and Visualization:
        - Tools to look at feature importance, errors, and progress over time (`showing_feature_noams`, `showing_residuals`, `showing_learn_curve`).
        - Specific graphs showing how features affect each other (`showing_limited_dependency`).
2. Testing Setup (`gbt_testing.ipynb`):
    - This file checks that everything works as it should, including:
    - Model Consistency: Makes sure the model behaves as expected (for example, how it formats outputs and calculates errors).
    - Hyperparameter Effects: Tests how changing `num_est`, `learn_quan`, and `high_dep` affects training and results.
    - Graph Tests: Confirms that graphs (like errors and learning curves) are created properly.
    - It uses fake and challenging datasets to test the model’s strength.

3. Usage Notebook (`gbt_application.ipynb`):
    - This shows how to use the model from start to finish:
    - Data Management: Loads and prepares data from `sample_data.csv`.
    - Model Training: Fits the GBT model to the training data.
    - Performance Check: Figures out scores like Mean Squared Error (MSE) and R² for both training and testing data.
    - Visual Learning:
    - Scatter plots that show real vs. predicted values.
    - Residual plots to find patterns in the errors.
    - Feature importance and limited dependency analysis to understand results better.

4. Sample Data and Extra Scripts (`sample_data.ipynb` and `sample_data.csv`):
    - `sample_data.csv`:
        - A CSV file that contains made-up regression data with features and labels, made for showing and testing the model.
    - `sample_data.ipynb`:
        - A notebook explaining how the sample data was made, including:
        - Creating random data to control how features relate to each other.
        - Adding noise to see how the model performs in real life.

# Evaluation Metrics:
Mean Squared Error (MSE):
    MSE is measured during each step of training the model. It checks how close the predictions are to the real values. The function `perform_r2()` calculates the leftover errors to figure out MSE. This function is also used in `gbt_testing.ipynb` for checking performance.
R-squared (R²)*
    The R² score is found using the `perform_r2()` method. This score shows how well the model explains the differences in the data. An R² value close to 1 means the model makes accurate predictions.
 
```python
def perform_r2(self, act_y, est_y):
tot_s2 = np.sum((act_y – np.mean(act_y)) ** 2) 
res_s2 = np.sum((act_y – est_y) ** 2) 
return 1 – (res_s2 / tot_s2)
```

This function calculates the total difference (total sum of squares) and the leftover difference (residual sum of squares) to get the R² score.
# Learning Curve:
To make sure the model learns correctly and doesn’t make mistakes by being too simple or too complex, a learning curve is drawn. This shows how the Mean Squared Error goes down as more trees are added to the model. This is done in the `showing_learn_curve()` function to see how well the model is training.

```python
def showing_learn_curve(self):
plt.figure(figsize=(9, 5))
plt.plot(range(1, self.num_est + 1), self.train_residuals, color=’blue’)
plt.xlabel(“Number of Estimators”)
plt.ylabel(“MSE”)
plt.title(“Learning Curve Visualization”)
plt.show()
```
# Residuals Plot:
The residuals plot comes from the `showing_residuals()` function. This plot shows the differences between what the model predicted and the actual values. It helps identify any consistent errors and shows where the model can be better. 

```python
def showing_residuals(self, X, y): 
estimators = self.predict(X) differences = y – estimators
plt.figure(figsize=(9, 4))
plt.scatter(estimators, differences, alpha=0.7, color=’green’)
plt.axhline(y=0, color=’red’, linestyle=’—‘) 
plt.xlabel(“Predictions”)
plt.ylabel(“Differences”)
plt.title(“Residuals Plot”)
plt.show() 
```
# Implementing and how to use the Boosting tree model:
To use the Gradient Boosting Tree (GBT) model from this project, just follow these simple steps:
1. Get the Project Files:
  First, download the project files to your computer or open Google Colab: 
Bash
 git clone https://github.com/sravanyabezawada/Project2.git cd Project2 
2. Upload Files to Google Colab:
    If you are using Google Colab: 
        - Upload these files: `gbt.py`, `gbt_testing.ipynb`, `gbt_application.ipynb`, `sample_data.ipynb`, and `sample_data.csv`.
        - Open `gbt_application.ipynb` to work with a sample dataset. 
3. Install Required Libraries:Make sure you have the necessary libraries:
 
 ```python
 !pip install numpy pandas matplotlib
 ``` 
 4. Run the Model:
    To use the GBT model, open the `gbt_application.ipynb` notebook. It will guide you on how to apply the model to the sample data (`sample_data.csv`). 

# Here’s how the notebook is put together: 
1. Load the Data: The sample data (`sample_data.csv`) is loaded, which includes example features and target values. 

 ```python
 import pandas as pd data = pd.read_csv(‘sample_data.csv’)
 X = data[[‘Feature1’, ‘Feature2’, ‘Feature3’]].values
 y = data[‘Target’].values
 ```

2. Divide Data into Training and Testing Sets: The data is divided into a training set (80%) and a testing set (20%). 

```python
from sklearn.model_selection 
import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
``` 

3. Create and Train the Model:
    - The `GBT` model is created with your chosen settings, like learning rate, number of trees, and tree depth. 

```python
 from gbt import GBT model = GBT(learning_rate=0.1, n_estimators=50, max_depth=3) model.fit(X_train, y_train) 
```

4. Make Predictions:  
    - After training, you can predict using the test set. 
```python
python predictions = model.predict(X_test)
```

5. Check Model Performance: 
You can check how well the model did using Mean Squared Error (MSE) and R² score. 

 ```python
mse = np.mean((y_test – predictions) ** 2)
r2 = model.perform_r2(y_test, predictions) 
print(f’Mean Squared Error: {mse}’) print(f’R² Score: {r2}’) 
 ```

6.Vi sualize Results:
You can also create visual representations of the training vs testing results, differences in predictions, and the importance of each feature using the methods in the `GBT` class. 

7. Adjusting the Model: You can change the model settings in the `GBT` class:
 - `learning_rate`: This sets how fast the model learns from past.
 - `n_estimators`: This is the number of trees to train.
 - `max_depth`: This limits how deep each tree can go. 

For example: 
```python
python model = GBT(learning_rate=0.05, n_estimators=100, max_depth=4)
``` 

 6. Visuals and Analysis- 
Scatter Plot of Train vs Test: See how the model performs on training and testing data.
 - Residual Plot: Shows the differences between what was predicted and the actual values. 
 - Feature Importance: Shows how important each feature is for predictions. Use the functions in `gbt.py` to create these visuals and analyze the model’s performance: 
 
```python
model.scatter_train_test(X_train, y_train, X_test, y_test) model.showing_residuals(X_test, y_test) model.showing_learn_curve()
``` 

 These steps will help you load data, train the model, adjust settings, and check how well the model performs easily.


## 1.	What does the model do and when should you use it? 
The model used in this project is called a Gradient Boosted Tree (GBT). This method helps make predictions by combining many simple models called decision trees. Each tree corrects mistakes from the ones before it. The goal of the GBT model is to reduce prediction mistakes by adding trees that improve the results step by step.
You should use this model: 
    - Getting accurate predictions is very important: as GBT usually works better than other models like linear regression or random forests on complex data. 
    – You need to deal with complicated relationships between the input features and the outcome you are trying to predict, which makes it great for problems where understanding how the features interact matters.
    – Your dataset is small to medium-sized, since GBT can make mistakes if it tries to fit too many trees or if they are too deep without proper adjustments. 
For example, we  have created a sample data where the target value `y` is based on a mix of features `X` with a bit of added randomness: 

 ```python
 X = np.random.rand(100, 3) y = 3 * X[:, 0] + 5 * X[:, 1] – 2 * X[:, 2] + np.random.normal(0, 0.1, 100)
 ```
 
In this case, the GBT model is used to predict `y` from the features `X`. It shines when there are complicated interactions between the features, as shown by its ability to fit the data accurately even with some noise.
In short, GBT is a strong option for making predictions when you need high accuracy, and it can be used in many areas like financial forecasting, marketing predictions, and risk analysis. 

## 2.How did you test your model to determine if it is working reasonably correctly? give me good answer? 
We have checked the GBT (Gradient Boosted Trees)model by following few simple steps to see how well it performed. This involved separating the data into two parts. They are  *Mean Squared Error (MSE)* and *R² Score* to evaluate it, and looking at plots to get a an idea of its predictions.
1. Train-Test Split: To make sure the model could work well on new data, I split the dataset into two parts: 80% for training and 20% for testing 

 ```python
 split_training_ratio = 0.8 sp_train = int(len(X) * split_training_ratio)
 train_x, test_x = X[:sp_train], X[sp_train:]
 train_y, test_y = y[:sp_train], y[sp_train:] 
 ```

2. Training the Model: 
We have trained the model using the `fit` method. This uses the training data to adjust the model gradually: 
 
```python
model = gbt(num_est=50, learn_quan=0.1, high_dep=3)
model.fit(train_x, train_y) 
```

3. Mean Squared Error (MSE): 
After training, We have calculated the MSE on the test data to see how far off the predictions were:

```python
mse = np.mean((test_y – estimators) ** 2)
print(f”Test MSE: {mse}”) 
```
A lower MSE mean the predictions are closer to the real values, showing better model performance. 

4. R² Score: 
We have also looked at the R² score to find out how well the model explained the changes in the target variable: 

```python
r2 = model.perform_r2(test_y, estimators) 
print(f”Test R²: {r2}”)
``` 

The R² score goes from 0 to 1, where a higher score means the model fits the data better. 

5. Visual Inspection: 
We have created plots to represent the differences between predicted and actual values, scatter plots are used to check how the model was Performing visually.

```python
model.showing_residuals(test_x, test_y) model.scatter_train_test(train_x, train_y, test_x, test_y)
``` 

6. Learning Curve:
Lastly, We had plotted the learning curve to represent how the MSE changed with the number of trees. This helped us to confirm that the model improved when we added more trees: 

 ````python
 model.showing_learn_curve()
 ```` 
These all together gave me a clear picture of how well the model was performing and it was doing its job effectively.

## 3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.) 
In our Gradient Boosted Tree (GBT) model, we’ve made some options available for users so that they can change to improvise the code for their requirements: 
1.`num_est` (Number of Trees): This command determines how many decision trees will be built while boosting. If you increase the number of trees, the model usually becomes more accurate, but it also takes more time to compute. More trees can make the model fit better, but if there are too many, it might not perform well on smaller datasets. 

```python
model = gbt(num_est=50, learn_quan=0.1, high_dep=3) 
```
2. `learn_quan` (Learning Rate): The learning rate decides how much each new tree helps fix the mistakes of the previous trees. A lower learning rate means the model learns more slowly, which can help fit the data better and reduce the risk of overfitting. But, if you set a lower learning rate, you might need more trees (`num_est`) to get good results.

```python
model = gbt(learn_quan=0.05) 
```
3. `high_dep` (Max Depth of Trees): This command controls how deep each decision tree can grow. Deeper trees might find more complex patterns, but they can also overfit the data if there are too many trees. By limiting the depth, you can help prevent overfitting and speed up training. 

 ```python
 model = gbt(high_dep=3) 
 ```
These let users adjust the model’s performance. By changing these values, users can find a good balance between how complex the model is, how long it takes to train, and how accurate it is. 
For example, a user could set:

 ```python
 model = gbt(num_est=100, learn_quan=0.1, high_dep=5)
 model.fit(train_x, train_y) 
 ```
This sets the model to train with 100 trees, a learning rate of 0.1, and a maximum tree depth of 5. This setup can help balance good performance and avoid overfitting. 
Being able to adjust these settings makes the model flexible for different datasets and tasks, allowing users to try out various setups to get the best results.

## 4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
In this project, the GBT model can have problems with these types of inputs:
- Large Datasets: The model builds decision trees one at a time, whichh can make it slow when using large datasets. Each tree needs to be updated based on the previous one, making the training process take longer. For instance, using a big dataset like this can cause delays or memory issues: 

 ```python
 X = np.random.rand(1000, 3) # A bigger dataset with 1000 samples
 y = 3 * X[:, 0] + 5 * X[:, 1] – 2 * X[:, 2] + np.random.normal(0, 0.1, 1000)
 ```
- Possible Fix: If we had more time, we could split the tree-building work into smaller parts so it runs faster or use a method that randomly samples the data to make each tree, which would lighten the workload. 
- Noisy or Unrelated Features: The model might also struggle if the dataset has distracting or unhelpful features. For example, if some features do not relate to the target variable, the model might focus on noise instead, leading to poor results on new data. 

 ```python
 X = np.random.rand(100, 10) # A dataset with more features, some irrelevant
 y = 3 * X[:, 0] + 5 * X[:, 1] – 2 * X[:, 2] + np.random.normal(0, 0.1, 100)
 ```
- Possible Fix: We could use methods to pick better features or control tree growth to help the model not get misled by unhelpful data.If there are too many irrelevant features, we might need to use more advanced ways to cut them down. 
- Dealing with Outliers: If the dataset has extreme values, the model might not perform well because decision trees can be affected by these extremes. For example, having some strange values like this could cause issues: 

```python
X = np.random.rand(100, 3)
 y = 3 * X[:, 0] + 5 * X[:, 1] – 2 * X[:, 2] + np.random.normal(0, 0.1, 100) 
y[0] = 1000 # Adding an outlier 
```
- possible Fix: We could find and handle outliers before using the data, or use a different loss method that is less sensitive to extreme values to help reduce their effect on building trees.In short, while these problems are not basic flaws in the GBT model, they are challenges we can tackle with some extra steps. For large datasets, splitting up the work is a good solution, while for noisy or unrelated data, selecting features and controlling tree growth can make the model stronger.

### Conclusion from the Results After running the code and looking at the results and charts, you can say that: 
1. The model works well, with low MSE and high R². 
2. The scatter plots show that the predictions are very accurate in both training and testing. 
3. The learning curve indicates that the model gets better as more estimators are added, and there is no sign of overfitting. 
4. The feature importance plot points out which features are the most important for predicting the target variable.



