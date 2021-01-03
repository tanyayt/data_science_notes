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

  


# Resource and Ref

[Pipeline on Kaggle](https://www.kaggle.com/alexisbcook/pipelines) 