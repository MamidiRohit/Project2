# Project 2

## Model Selection

Implement generic k-fold cross-validation and bootstrapping model selection methods.

In your README, answer the following questions:

* Do your cross-validation and bootstrapping model selectors agree with a simpler model selector like AIC in simple cases (like linear regression)?
- Regarding simple data sets, it can be confirmed that the previous two models and simpler models such as AIC reach the same conclusion. However, when testing complex datasets that are difficult to estimate (e.g., with many random elements) or when testing with multi-collinearity in mind, the results are somewhat inconsistent. 
- 
* In what cases might the methods you've written fail or give incorrect or undesirable results?
- K겹 교차검증의 경우, 다중공선성이 포함되면  
* What could you implement given more time to mitigate these cases or help users of your methods?
* What parameters have you exposed to your users in order to use your model selectors.
