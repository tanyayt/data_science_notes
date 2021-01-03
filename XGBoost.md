# XGBoost

Ensemble methods combine the predictions of several models. Gradient boosting is another ensemble method. 

## Gradient Boosting

Gradient boosting is a method that iteratively adds models into an ensemble: 

1. Initialize the ensemble with a simple model
2. Use the ensemble to generate predictions. To make a prediction, add the predictions from all models in the ensemble
3. Calculate a loss function
4. Use the loss function to fit a new model, that will be added to the ensemble. Specifically, determine new model's parameters so that new model will reduce the loss 
5. Add new model to ensemble and repeat 

<img src="https://i.imgur.com/MvCGENh.png">

XGBoost stands for extreme gradient boosting

```python
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)
```

## Parameter Tuning

XGBoost has a few parameters than can affect accuracy and training speed

- `n_estimators` specifies # of times to iterate the cycle. It is equal to the number of models we include in the ensemble; if n_estimators is too low, we may underfit, if too high, we might overfit; typical values range from 100-1000

* `early_stopping_rounds` early stopping stops the model from iterating, when the validation score stops improving. It's smart to set a high value for n_esitmator, and have early_stopping_rounds find the optimal time to stop iterating. If we set early_stopping_rounds = 5, we stop after 5 straight rounds of deteriorating validation scores (avoid randomly having 1 round not improving) 

* ### `learning_rate`

  Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.

  This means each tree we add to the ensemble helps us less. 

```python
my_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```

* `n_jobs`: On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on your machine. On smaller datasets, this won't help.



## Reference 

[XGBoost@Kaggle](https://www.kaggle.com/alexisbcook/xgboost)