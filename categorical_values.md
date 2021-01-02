# Categorical Values 


```
# Get list of categorical variables 
#(note the code below returns all object columns, which can be str or mix 
import pandas as pd
object_cols = df.select_dtypes(include='object').columns

```

## Label Encoding
Convert to ordinal variables (e.g. everyday, sometimes, never to 3,2,1) 
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

```
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
# We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data
# setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

```
