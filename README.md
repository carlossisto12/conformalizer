# Conformalizer



This repository aims to provide tools for Data Science enthusiasts who seek to measure the uncertainty of the predictions generated by their regression and classification algorithms, using Conformal Predictions.



## Conformal Predictions

Conformal Predictions can be used in regression or classification problems, assuming independet and identically distributed data. In regression problems, an confidence interval for the predicitions is built, while in classification, a set of values is returned, both with $(1-\alpha)$ confidence level. This method guarantee this level of significance, for more informations see [[1](#1)].

To compute this metric, we need to divide the dataset in train, calibration and test subsets. Than the following step-by-setep is executed, further details can be seen in [[1]](#1):

- Firts of all, an uncertainty metric must be defined. Like the predicted probability of an instance belongs to each class in a classification problem.
- Define the score function that calculates how distant a predicted value is from the true one..
- Use a model fitted with the train subset, calculate the score values in the calibration data;
- Calculate the quantile $\hat{q}=\frac{\left[\left(n+1\right)\left(1-\alpha\right)\right]$;
- Use this quantile to obtain the confidence sets in test set.

## Conformal Predictions in classification

Conformalizer library uses the probability of an instance belongs to each class as score function. Two different methods are implemented: greedy and non-greedy. This method can be found in sections 1.1 and 2.1 in [[1](#1)].

To compute Conformal Predictions using the conformalizer library, just use the ``clssification_conformal`` method. It has the ``ClassificationConformal`` class that has the following parameters:

- ``df``: Complete DataFrame to be used in the model;
- ``target``: Name of the target column in df;
- ``n_calib``: Number of observations to be used in calibration;
- ``random_state``: Seed to be used in sets division (default=42);
- ``test_size``:  Proportion of data to be used in test, excluding calibration set size (default=0.2);
- ``scaler``: sklearn scaler object. If None, the values will not be scaled (default=None);

Than the conformal sets can be calculated in two ways, using a XGBoost model or any sklearn classifier that has the ``predict_proba`` method.

To use a XGBoost model, use the ``xgb_conformal_classification`` method. It will return a DataFrame with the true values in test set, the conformal set and its confidence. It has the following parameters:

- ``alpha``: Significance level of classification sets;
- ``parameters``: Dictionary with the parameters to be used in XGBoost model. If None a dictionary containing the ``softprob`` objective and ``num_classes`` as the number of classes is set (default=None);
- ``num_boost_round``: Boosting rounds in XGBoost training (default=30);
- ``method``: Which conformal predicition method to be used, can be "standard" for non-greedy, or "adaptive" fro greedy.

To use sklern classifiers, the ``conformal_classification``method must be used. It will return a DataFrame with the true values in test set, the conformal set and its confidence. It has the following parameters:

- ``alpha``: Significance level of classification sets;
- ``classifier``: sklearn classifier, as ``RandomForestClassifier()``;
- ``method``: Which conformal predicition method to be used, can be "standard" for non-greedy, or "adaptive" fro greedy.

### Conformal Predictions in Regression

The conformalizer module can use Linear Quantile Regressions, Random Forests or Gradient Boosting algorithms to generate confidence intrevals for predictions, and the score function used is $\max{\hat{t}_{\alpha/2}(x) - y,y -\hat{t}_{1-\alpha/2}(x)\}$, for mor details, the method can be found in the section 2.2 in [[1](#1)].

To compute Conformal Predictions for regression, use the ``regression_conformal`` module. It has the the ``ConformalRegression`` classe, which has the the same paramters the ``ConformalClassification`` class. All of its methos returns a DataFrame with the true value in test set and columns with the lower and upper bounds of predicted interval.

To use linear quantile regression to compute the conforml predictions, the method ``linear_conformal_regression`` must be used. It has two parameters:

- ``formula``: Formula to be used in regression;
- ``alpha``: Significance level of conformal intervals.

To use random forest to compute the conformal predictions, the method ``random_forest_conformal_regression`` must be used. It has two parameters:

- ``parameters``: Dictionary with the parameters to be used in ``RandomForestRegressor``. If None, the default values are used (default=None).
- ``alpha``: Significance level of conformal intervals.

To use gradient boosting to compute the conformal predictions, the method  ``gradient_boosting_conformal_regression`` must be used. It has two parameters:

- ``parameters``: Dictionary with the parameters to be used in ``GrdientBoostingRegressor``. If None, the defult values are used (default=None).
- ``alpha``: Significance level of conformal intervals.



### Notes

Conformal Predictions are powerfull tools, but don't fix the problems of use a model that can't explain your data. If it's happen, you will get the prediction sets or intervals, but the results will not be trustworth.

### References

[1] ANGELOPOULOS, A. N.; BATES, S. A gentle introduction to conformal prediction and
distribution-free uncertainty quantification. CoRR, abs/2107.07511, 2021. Available in: <https:
//arxiv.org/abs/2107.07511>.
