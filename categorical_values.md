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
