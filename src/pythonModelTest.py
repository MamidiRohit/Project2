import numpy as np
from sklearn.linear_model import LinearRegression
from modelSelection import ModelEvaluations
# generate some synthetic data
X = np.random.rand(100, 3)
y = X @ np.array([2, -1, 3]) + np.random.normal(0, 0.5, 100)

# create an instance of the LinearRegression model
model = LinearRegression()
# call the kFoldCv method
kFoldMseScores, kFoldAvgMse = ModelEvaluations.kFoldCv(X, y, model, k=5)
print("k-Fold MSE Scores:", kFoldMseScores)
print("Average MSE from k-Fold:", kFoldAvgMse)

bootstrapMseScores, bootstrapAvgMse = ModelEvaluations.bootstrap(X, y, model, numIterations=100)
print("Bootstrap MSE Scores:", bootstrapMseScores)
print("Average MSE from Bootstrapping:", bootstrapAvgMse)

aicValue = ModelEvaluations.calculateAic(X, y, model)
print("AIC Value:", aicValue)
