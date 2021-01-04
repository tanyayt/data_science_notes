# Data Leakage 

Data leakage occurs when training data contains information about the target, leading to high performance on the training set while low performance in production. Two types of leakage: target leakage, and train-test contaminaiton

## Target Leakage 

Predictors include data that will not be available at the time you make predictions. e..g predict who get sick with penumonia; People take antibiotics after getting pneumonia. This is target leakage. 

e.g. In house price prediction, if avg price in the neighborhood is updated after the home is sold, then this creates a target leakage when using avg price in the neighborhood as a feature. 

## Train-test contamination

If validation data affects the preprocessing steps. 



## Identifying Data Leakage 

Things to ask: how were data collected? 

## Reference 

[Data Leakage @Kaggle](https://www.kaggle.com/alexisbcook/data-leakage)

