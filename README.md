# StepwiseSelectionOLS
Best Subset Selection, Forward Stepwise, Backward Stepwise Classes in sk-learn style.

This package is compatible to sklearn. Examples on `Pipeline` and `GridSearchCV` are given.

## ForwardStepwiseOLS

2020-04-19

**Hyperparameter**

`fK`: at most `fK` number of features are selected

## BackwardStepwiseOLS

2020-04-29

**Hyperparameter**

`fK`: at least `fK` number of features are selected, $fK>=1$

## BestSubsetOLS

**Hyperparameter**

`fK`: exactly `fK` number of features are selected

# Reference

https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py

https://en.wikipedia.org/wiki/Stepwise_regression