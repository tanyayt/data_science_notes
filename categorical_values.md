# Categorical Values 


```
# Get list of categorical variables 
#(note the code below returns all object columns, which can be str or mix 
import pandas as pd
object_cols = df.select_dtypes(include='object').columns

```

Cardinality: the number of unique entries of a categorical variable is the cardinality of the variable 

## Label Encoding

Convert to ordinal variables (e.g. everyday, sometimes, never to 3,2,1) ;

Drawback: the values in train data may not cover all values in the test data

```
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])

```



## One Hot Encoding
One-hot encoding creates new columns indicating the presence (or absence) 
of each possible value in the original data.
<img src="https://i.imgur.com/TW5m0aJ.png"> 
Unlike label encoding, one hot encoding does not assume ordering of categories. One-hot encoding does NOT perform well with large number of categorical values

```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
# We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
# setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

# OneHotEncoder returns a 2D array after fit_transform so we need pd.DataFrame() to turn it back to pandas dataframe

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

```

Typically, we only use one-hot encoding to encode columns with low cardinality. The high cardinality columns can either be dropped or we can use label encoding instead. 

```python
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
```



### Use `pd.get_dummies()`

```python
X_train = pd.get_dummies(X_train)
```



