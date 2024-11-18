

**1. What does the model you have implemented do, and when should it be used?**

The Boosting Tree Model implements a Gradient Boosting algorithm for regression tasks. Gradient Boosting works by training a sequence of decision trees iteratively, where each tree aims to correct the errors made by the previous ones.  The model begins with an initial prediction, typically the mean of the target variable, and refines this over multiple iterations. At each iteration, it calculates residuals (errors) and trains a new decision tree to predict them. The predictions from this tree are scaled by the learning rate and added to the cumulative predictions, progressively improving accuracy. 

Each decision tree is a weak learner, shallow and focused on correcting specific residuals. Together, they form a strong model capable of handling feature interactions and capturing complex, nonlinear relationships. The boosting process continues for a set number of iterations (num_trees) or until error improvement falls below a threshold (tol), minimizing the Mean Squared Error (MSE). This Gradient Boosting implementation approximates the gradient of the loss function by learning from residuals at each step. It is robust, handling data normalization, missing values, and sparse data, making it effective for regression tasks on structured datasets with complex patterns. 

The Boosting Tree Model works best on nonlinear data and complex patterns, as shown in the test cases. It struggles with linear data but performs well on high-dimensional and noisy datasets, making it ideal for regression tasks with structured data. 

**2. How did you test your model to determine if it is working reasonably correctly?** 

We tested the model using several synthetic datasets that we learned in class and real-world data, designed to challenge the model in different ways (we used the California Housing dataset, the “small_dataset” provided for the Project1, the Energy Efficiency Dataset and the Wine Quality Dataset). We also use the pre-built Scikit-learn Boosting Tree to compare the results with our model: 

Test Cases: 

**Linear Data:** The model struggled with simple linear patterns, showing low R2 scores compared to Scikit-Learn’s implementation. This highlighted its limitations in handling purely linear relationships. 

**Nonlinear Data:** The model performed well here, achieving high R2 scores, proving its strength in capturing complex patterns. 

**Collinear and High-Dimensional Data:** Performance was robust, but not as accurate as Scikit-Learn’s due to the lack of built-in regularization like shrinkage. 

**Extreme Scenarios (Sparse, Noisy Data):** The model performed reasonably well but required significant computation time, highlighting the need for further optimization. 

**Evaluation Metrics:**

R²: Measured how well the model explained variance in the data. Nonlinear datasets showed higher R2, while linear ones were lower. 

Mean Squared Error (MSE): Quantified prediction error. 

**Key Issues:** 

Low R2 on linear data suggested the model was not leveraging its full potential. However, on complex datasets, it significantly improved, matching or approaching Scikit-Learn’s performance. 

**3. What parameters have you exposed to users of your implementation to tune performance?** 
 
During development, we tested various ranges of the model’s parameters but settled on the following standard values: number of trees = 50, learning rate = 0.1, max depth = 3, subsample = 0.8, and tolerance = 1e-5. In some test cases, increasing the number of trees from 50 to 1000 led to improvements in R2. Similarly, raising the learning rate from 0.1 to 0.2 showed slight gains in accuracy. However, we left the standard as they provide a good balance between performance and runtime, according to the literature, and users can adjust them as needed for specific datasets. 

Here are the hyperparameters users can modify to optimize the model’s performance: 

**Number of Trees (num_trees):** Controls the number of boosting iterations. Higher values improve accuracy but can lead to overfitting and increased runtime. 

**Learning Rate (learning_rate):** Determines how much each tree contributes to the final prediction. Lower rates improve generalization but require more trees. 

**Maximum Tree Depth (max_depth):** Limits the complexity of each tree. Shallower trees reduce overfitting and speed up training. 

**Subsample Ratio (subsample):** Specifies the fraction of data used for each tree, adding randomness to improve generalization. 

**Tolerance (tol):** Sets an early stopping criterion based on the minimum error improvement, saving computation time. 

**4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these, or is it fundamental to the model?** 

We ran into a challenge while testing the model. First, sparse data was tough to handle. Initially, some splits ended up empty, causing issues with predictions. We fixed this by adding extra checks to ensure splits were not empty, but performance was still sensitive to sparsity. Another challenge was highly correlated features. While the model can handle multicollinearity to some extent, extreme correlation between features lowered performance. This showed up in our collinear test cases, where R2 scores were noticeably lower compared to Scikit-Learn’s gradient boosting. Lastly, large datasets made training much slower because each tree is built one at a time, increasing the overall computation time. 

With more time, we could make several improvements to handle these challenges better. For sparse data, we could explore better split strategies or even pre-process the data differently. For correlated features, introducing some form of regularization might help simplify the trees and reduce overfitting. Training on large datasets could be sped up by adding parallel processing for tree construction. 

Another area to explore would be fine-tuning the model to work better on linear data. Although simpler models like linear regression are a better fit for these cases, we noticed that Scikit-Learn’s gradient boosting performed surprisingly well even on linear datasets. This shows there’s potential to tweak our implementation to improve accuracy in those scenarios as well. 

**References** 

The Elements of Statistical Learning (Chapter 10, Boosting) 

From Mining of Massive Datasets (Chapter 10, Gradient Descent) 

