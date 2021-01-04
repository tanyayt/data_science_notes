# Deep Learning

## Linear Units in Keras

Sequential class groups a linear stack of layers into a tf.keras.Model; Sequential provides training and inference features on the model;

Dense refers to 1 layer; Sequential class can wrap multiple layers; each layer can wrap in multiple units 

```python
from tensorflow import keras 
from tensorflow.keras import layers 

# Create a network with 1 linear unit 
# units define how many outputs we want
# input_shape defines the dimensions of inputs, 3 features in this case
model = keras.Sequential([
     layers.Dense(units=1,input_shape=[3])  # note, input_shape must be a list
])
```

Keras represents the weights (`model.weights`)of a neural network with tensors. Tensors are TensorFlow version of NumPy arrays but in a way more suited to deep learning. Tensors are compatible with GPU and TPU (Tensor Processing Units) accelerators. 

## Stacking Dense Layers 

<Img src="https://i.imgur.com/Y5iwFQZ.png">

A layer can be thought as data transformation, e.g. convolutional layers. 

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),#second layer no need to specify input_shape 
    # the linear output layer 
    layers.Dense(units=1), #no activation so just linear 
])
```



## Activation Functions

Without activation functions, neural networks can only learn linear relationships. 

Relu function tends to do well in most problems; so may start from relu 

## Adding the loss and optimier 

After defaining the model we can add optimizer and loss function

```python
model.compile(
    optimizer = 'adam',
    loss = 'mae'
)
```

## Train the model 

```python
history = model.fit(
    X_train, y_train, 
    validation_data = (X_valid,y_valid),
    batch_size = 256, 
    epochs = 10
)

# stores the history object in a pandas dataframe 
history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
```

Smaller batch size may give noisier weight updates and loss curves. 



## Reference 

- [Intro to Deep Learning - Kaggle](https://www.kaggle.com/learn/intro-to-deep-learning) 