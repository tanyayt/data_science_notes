# Convolutional Classifier 

A convnet consists of two parts: 

- A convolutional base: to extract the features from an image. It's formed primarily of layers performing the convolutional operation 

- A dense head: formed primarily of dense layers, and can include dropout ; to determine the class of the image 

  <img src = "https://i.imgur.com/U0n5xjU.png"> 

  <img src="https://i.imgur.com/UUAafkn.png">

  

## Training the classifer 

Two goals in netowrk training: 

1. which features to extract from an image (base)
2. which class goes what features (head)

We rarely train a model from scratch but we reuse the base of a pretrained model, and then attach an untrained head. Reusing a pretrained model is a technique known as transfer learning. It is so effective, that almost every image classifier these days will make use of it. 

```python
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# load pretrained base 
# e.g. the most commmonly used pretaraining source is ImageNet
pretrained_base = tf.keras.models.load_model(
    'some_pretrained_base'
)
pretrained_base.trainable = False 
# attach head
model = keras.Sequential([
     pretrained_base,
     layers.Flatten(),#transforms 2D outputs of the base into 1D inputs for head in the next layer 
     layers.Dense(6,activation='relu'),
     layers.Dense(1,activation='sigmoid')
])

#train 
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
)

history = model.fit(
    ds_train,
     validation_data = ds_valid,
     epochs =30
)

```

