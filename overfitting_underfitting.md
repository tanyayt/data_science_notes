# Overfitting & Underfitting 

The loss over epochs plot is called a learning curve 

<img src="https://i.imgur.com/tHiVFnM.png">

- Underfitting the training set is when the loss is not as low as it could be because the model hasn't learned enough signal
- Overfitting the training set is when the loss is too low as it could be because the model learned too much noise 
- Capacity: a model's capacity refers to the size and complexity of patterns it is able to learn. e.g. in neural networks, you can increase the capacity of a network by making it wider or deeper; wider networks can learn linear relationships better while deeper netowrks prefer more non-linear ones
- **Early stopping** we can stop the training when the validation loss insn't decreasing anymore; 

## Adding Early Stopping in Keras 

In Keras, we include early stopping in our training through a callback. A callback is just a function you want run every so often while the network trains. The early stopping callback will. run after every epoch. 

"These parameters say: "If there hasn't been at least an improvement of 0.001 in the validation loss over the previous 20 epochs, then stop the training and keep the best model you found." It can sometimes be hard to tell if the validation loss is rising due to overfitting or just due to random batch variation. The parameters allow us to set some allowances around when to stop."

```python
from tensorflow.keras.callbacks import EarlyStopping

# define early stopping criteria
early_stopping = EarlyStopping(
    min_delta=0.001, # minimum change to count as improvement 
     patience = 20, # how many epochs to wait before stopping 
     restore_best_weights = True #we keep the model where the validation loss was lowest 
)

# define model architecture
model = keras.Sequential([
     layers.Dense(512,activation = 'relu', input_shape = [11]),
     layers.Dense(512,activation = 'relu'),
     layers.Dense(512,activation = 'relu'),
     layers.Dense(1)
])

# define optimizer and loss function
model.compile(
    optimizer = 'adam',
    loss = 'mae'
)

# fit the model with training data, and early_stopping criteria 
history = model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 256, 
    epochs = 500,
    callbacks = [early_stopping], #put your callbacks in a list 
    verbose = 0 #turn off traning log 
)

# store training history in data frame for plotting 
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
```

## Dropout

* Overfitting is caused by the model learning noise (rather than signal), as it shows in the model is that the network often rely on a very specific combinations of weights, a kind of conspiracy of weights. Being so specific, they tend to be fragile: remove on and conspiracy falls apart
* This is the idea behind dropout. We randomly drop out some faction of a layer's inputs in every step of training to make harder for the model to pick up spurious patterns.It has to learn general patterns that tend to be more robust 

```python
keras.Sequential([
     #....some other layers above
     layers.Dropout(rate = 0.3) # apply 30% dropout to the NEXT layer 
     layers.Dense(16)
])
```



## Batch Normalization

Aka, batchnorm, which can help training that is too slow or unstable; you can put the batchnorm after a layer.

Normalization can be part of the preprocessing with StandardScaler or MinMaxScaler. Because SGD shifts network weights in proportion to how large an activation the data produces. Features that tend to produce activations of very different sizes can make for unstable training behavior 

Now, if it's good to normalize the data before it goes into the network, maybe also normalizing inside the network would be better! In fact, we have a special kind of layer that can do this, the **batch normalization layer**. A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, and then also putting the data on a new scale with two trainable rescaling parameters. Batchnorm, in effect, performs a kind of coordinated rescaling of its inputs.

```
layers.Dense(16, activation = 'relu'),
layers.BatchNormalization(),	
```

or between a layer and its activation function:

```python
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu')
```

### Example with both dropout and batch normalization 

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```





## Reference/Resource 

- [Overfitting and underfitting @Kagge](https://i.imgur.com/tHiVFnM.png)
- [Dropout and batch normalization@Kaggle](https://www.kaggle.com/ryanholbrook/dropout-and-batch-normalization)