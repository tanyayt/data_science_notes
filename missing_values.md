# Handling Missing Values
## Resource/Reference: 
- Mssing Values [Kaggle](https://www.kaggle.com/alexisbcook/missing-values)

## Drop Columns with Missing Values 
- Cons: may loose a lot of important info
- When to use: when there are relatively few missing entries

Code Example: 
```
# save the names of columns that contain missing values 
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training set (can do the same on test set) 
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
```


## Imputation
- What: fill in missing values with some numbers, e.g. mean values of each column
- Problem: imputed values may be systematically above or below the actual values 

Code Example:
```
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer() #default strategy is mean; default missing value is mean; alternatively, can use median,                                # most frequent, or constant 
                             # default add_indicator = False; but can add a column indicate original/imputed values 

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))  # fit_transform() must be used before .transform()
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) # no need to fit again

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

## Extension to Imputation
- What: add a column to show the locations of imputed entries
