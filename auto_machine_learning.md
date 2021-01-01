# Auto Machine Learning

## Resource & Reference
1. Intro to AutoML course on Kaggle (https://www.kaggle.com/alexisbcook/intro-to-automl) 

## 7-Steps in Machine Learning
- Step 1: Gather the data (challenges: target leakage when training data contains info about the target, leading to high performance with training data but poor performance with test data) 
- Step 2: Prepare the data (dealing with missing values; feature engineering)
- Step 3: Select a Model 
- Step 4: Train the model
- Step 5: Evaluate the model 
- Step 6: Tune the parameters (e.g. tune parameters from XGBoost models)  
- Step 7: Get the predictions

AutoML automates step 2-7 (available on Google Cloud) 

## AutoML with Google Cloud
### What's needed
- `PROJECT_ID` - The name of your Google Cloud project. All of the work that you'll do in Google Cloud is organized in "projects".
- `BUCKET_NAME` - The name of your Google Cloud storage bucket. In order to work with AutoML, we'll need to create a storage bucket, where we'll upload the Kaggle dataset.
- `DATASET_DISPLAY_NAME` - The name of your dataset.
- `TRAIN_FILEPATH` - The filepath for the training data (train.csv file) from the competition.
- `TEST_FILEPATH` - The filepath for the test data (test.csv file) from the competition.
- `TARGET_COLUMN` - The name of the column in your training data that contains the values you'd like to predict.
- `ID_COLUMN` - The name of the column containing IDs.
- `MODEL_DISPLAY_NAME` - The name of your model.
- `TRAIN_BUDGET` - How long you want your model to train (use 1000 for 1 hour, 2000 for 2 hours, and so on).

### Code Example 
```
PROJECT_ID = 'kaggle-playground-170215'
BUCKET_NAME = 'automl-tutorial-alexis'

DATASET_DISPLAY_NAME = 'taxi_fare_dataset'
TRAIN_FILEPATH = "../working/train_small.csv" 
TEST_FILEPATH = "../input/new-york-city-taxi-fare-prediction/test.csv"

TARGET_COLUMN = 'fare_amount'
ID_COLUMN = 'key'

MODEL_DISPLAY_NAME = 'tutorial_model'
TRAIN_BUDGET = 4000

# Import the class defining the wrapper
from automl_tables_wrapper import AutoMLTablesWrapper

# Create an instance of the wrapper
amw = AutoMLTablesWrapper(project_id=PROJECT_ID,
                          bucket_name=BUCKET_NAME,
                          dataset_display_name=DATASET_DISPLAY_NAME,
                          train_filepath=TRAIN_FILEPATH,
                          test_filepath=TEST_FILEPATH,
                          target_column=TARGET_COLUMN,
                          id_column=ID_COLUMN,
                          model_display_name=MODEL_DISPLAY_NAME,
                          train_budget=TRAIN_BUDGET)

```

```
# Create and train the model
amw.train_model()

# Get predictions
amw.get_predictions()
```
