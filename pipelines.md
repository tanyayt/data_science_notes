# Pipelines

A pipeline combines preprocessing and modeling steps so you can use the bundle as a whole step 

Benefits of Pipelines: 

* Cleaner code 

* Fewer bugs 

* Easier to productionize 

* More options for model validation, e.g. cross validation 

## Steps

- Step 1: Define preprocessing steps with `ColumnTransformer` class: impute missing values in numerical columns, and applies one-hot encoding to categorical data

  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import OneHotEncoder
  
  # preprocessing for numerical data
  numerical_transformer = SimpleImputer(strategy='constant')
  
  # preprocessing categorical data
  categorical_transformer = Pipeline(steps = [
       ('imputer',SimpleImputer(strategy='most_frequent')),
       ('onehot',OneHotEncoder(handle_unknown='ignore'))
  ])
  
  # bundle preprocessing steps together
  preprocessor = ColumnTransformer(
      transformers=[
           ('num',numerical_transformer,numerical_cols),
           ('cat',categorical_transformer,categorical_cols)
      ]) #numerical_cols and categorical_cols are pre-defined
  
  #build model 
  from sklearn.ensemble import RandomForestRegressor
  model = RandomForestRegressor(n_estimators = 100, random_state=0)
  
  # bundle preprocessing and modeling steps in a pipeline 
  my_pipeline = Pipeline(steps = [('preprocessor',preprocessor),
                                  ('model',model)
                                 ] 
                        )
  
  #using pipeline: preprocess and fit model
  my_pipeline.fit(X_train,y_train)
  
  # pre_process and get predictions on validation data
  preds = my_pipeline.predict(X_valid)
  ```

  

## Example: make_column_transformer()

```python
spotify = pd.read_csv('../input/dl-course-data/spotify.csv')

X = spotify.copy().dropna() # drops any rows with missing values 
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num), # use standardscaler for numerical features 
    (OneHotEncoder(), features_cat), # use onehot encoder for categorical features 
)

# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100
```

* `GroupShuffleSplit()` GroupShuffleSplit first groups the sample set to be divided, and then divides the training set and test set according to the grouping. Train_size is the proportion of samples to be training set, .e.g. 0.8; 

# Resource and Ref

[Pipeline on Kaggle](https://www.kaggle.com/alexisbcook/pipelines) 