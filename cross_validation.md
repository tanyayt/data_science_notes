# Cross Validation

In cross-validation, we run our modellling process on different subsets of data to get multiple measures of model quality. e.g. we divide our data to 5 folds, and run one experiment on each fold

1. First fold as a validation set while rest is training data
2. Second fold as a validation set, while the rest is training data

When to Use: 

- Small dataset: yes, since extra computational burden is light 
- Large dataset: a single validation set may be sufficient 

```python

from sklearn.model_selection import cross_val_score

# set pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))

# Multiply by -1 since sklearn calculates *negative* MAE
# note, since we divide the dataset by number of folds (cv); no need to do train_test_split 
                              
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5, #set 5 folds 
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```

