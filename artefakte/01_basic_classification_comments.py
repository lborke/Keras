# Train your first neural network: basic classification
#
# This guide trains a neural network model to classify images of clothing, 
# like sneakers and shirts. It's okay if you don't understand all the details, 
# this is a fast-paced overview of a complete TensorFlow program with the details 
# explained as we go.
#
# This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
# https://www.tensorflow.org/tutorials/keras/basic_classification
# ==============================================================================


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


### Import the Fashion MNIST dataset
# ==============================================================================

# This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. 
# The images show individual articles of clothing at low resolution (28 by 28 pixels)

# We will use 60,000 images to train the network and 10,000 images to evaluate how accurately the network 
# learned to classify images. You can access the Fashion MNIST directly from TensorFlow, just import and load the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Loading the dataset returns four NumPy arrays:
#  - The train_images and train_labels arrays are the training set—the data the model uses to learn.
#  - The model is tested against the test set, the test_images, and test_labels arrays.
# The images are 28x28 NumPy arrays, with pixel values ranging between 0 and 255. The labels are 
# an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents

# Each image is mapped to a single label. Since the class names are not included with the dataset, 
# store them here to use later when plotting the images:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
# Let's explore the format of the dataset before training the model. 
# The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:
train_images.shape

# Likewise, there are 60,000 labels in the training set:
len(train_labels)

# Each label is an integer between 0 and 9:
train_labels

# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
test_images.shape

# And the test set contains 10,000 images labels:
len(test_labels)

# ==============================================================================


### Preprocess the data
# ==============================================================================

# The data must be preprocessed before training the network. If you inspect the first image in the training set, 
# you will see that the pixel values fall in the range of 0 to 255:
train_images[0][1:20,1:20]

# We scale these values to a range of 0 to 1 before feeding to the neural network model. 
# For this, we divide the values by 255. It's important that the training set and the testing set are preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

# ==============================================================================


### Build the model
# ==============================================================================

## Setup the layers

# The basic building block of a neural network is the layer. Layers extract representations 
# from the data fed into them. And, hopefully, these representations are more meaningful for the problem at hand.

# Most of deep learning consists of chaining together simple layers. Most layers, like tf.keras.layers.Dense, 
# have parameters that are learned during training.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels), 
# to a 1d-array of 28 * 28 = 784 pixels. Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has 
# no parameters to learn; it only reformats the data.

# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. 
# These are densely-connected, or fully-connected, neural layers. The first Dense layer has 128 nodes (or neurons). 
# The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1. 
# Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.

## Compile the model

# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
#  - Loss function —This measures how accurate the model is during training. We want to minimize this function to "steer" 
# the model in the right direction.
#  - Optimizer —This is how the model is updated based on the data it sees and its loss function.
#  - Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images 
# that are correctly classified.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==============================================================================


### Train the model
# ==============================================================================

# Training the neural network model requires the following steps:
#  1. Feed the training data to the model—in this example, the train_images and train_labels arrays.
#  2. The model learns to associate images and labels.
#  3. We ask the model to make predictions about a test set—in this example, the test_images array. We verify that 
# the predictions match the labels from the test_labels array.

# To start training, call the model.fit method—the model is "fit" to the training data:
model.fit(train_images, train_labels, epochs=5)

# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) 
# on the training data.

# ==============================================================================


### Evaluate accuracy
# ==============================================================================

# Next, compare how the model performs on the test dataset:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset. 
# This gap between training accuracy and test accuracy is an example of overfitting. Overfitting is 
# when a machine learning model performs worse on new data than on their training data.

# ==============================================================================


### Make predictions
# ==============================================================================

# With the model trained, we can use it to make predictions about some images.
predictions = model.predict(test_images)

# Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:
predictions[0]

# A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image corresponds to each 
# of the 10 different articles of clothing. We can see which label has the highest confidence value:
np.argmax(predictions[0])

# So the model is most confident that this image is an ankle boot, or class_names[9]. And we can check the test label 
# to see this is correct:
test_labels[0]

# Finally, use the trained model to make a prediction about a single image.

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. 
# So even though we're using a single image, we need to add it to a list:
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

# Now predict the image:
predictions_single = model.predict(img)
print(predictions_single)

# model.predict returns a list of lists, one for each image in the batch of data. Grab the predictions for our (only) image in the batch:
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
# ==============================================================================


