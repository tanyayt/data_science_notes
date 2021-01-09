# Classification

Before running neural network models, assign a 0/1 class label to the binary variable

## Accuracy and Cross-Entropy

Accuracy is one the many metrics in use for measuring success on a classification problem. Accuracy is `number_of_correct_predictions/total_predictions` The problem with accuracy and most other classification metrics is that, it can't be used as a loss function. SGD needs a loss function that changes smoothly; but accuracy is a ratio changes in jumps; so we have to choose something else: cross-entropy function

- In regression problems, we use MAE to measure the distance betweeen the expected outcome and the predicted outcome 
- In classification, we want a distance between probabilities. Cross-entroy is a measure for the distance from one probability to another. 

Cross-entropy and accuracy function both require probability as inputs in the range of 0-1. This is done by sigmoid activation. To get a final probability, we define a threshold probability, typically 0.5. Above 0.5 is going to be 1 in class label. 

## Example

```python
from tensorflow import keras
from tensorflow.keras import layers

# define model with relu and sigmoid in final output layer 
model = keras.Sequential([
     layers.Dense(4, activation = 'relu',input_shape=[33]), #only input layer needs input shape 
     layers.Dense(4,activation = 'relu'),
     layers.Dense(1,activation = 'sigmoid')
])

# add cross-entopy loss and accuracy metric to the model with compile 
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy'] # for binary/2 classes 
)

# define early stopping criteria 
early_stopping = keras.callbacks.EarlyStopping(
    patience = 10,
    min_delta = 0.001,
    restore_best_weights = True
)

# train model and save loss vs epoch in history 
history = model.fit(
    X_train,y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 512, 
    epochs = 1000, 
    callbacks = [early_stopping],
    verbose = 0 # hide output 
)

# to plot loss vs val_loss
history_df = pd.DataFrame(history.history)
# start the plot from epoch 5 
history_df.loc[5:, ['loss','val_loss']].plot()
history_df.loc[5:,['binary_accuracy','val_binary_accuracy']].plot()

print('Best validation loss: {:0.4f}' +"\nBest Validation Accuracy:{:0.4f}".format(history_df['val_loss'].min(),history_df['val_binary_accuracy'].max()))
```



