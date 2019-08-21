# Train your first neural network: basic classification
#
# This guide trains a neural network model to classify images of clothing, 
# like sneakers and shirts. It's okay if you don't understand all the details, 
# this is a fast-paced overview of a complete TensorFlow program with the details 
# explained as we go.

# This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories. 
# The images show individual articles of clothing at low resolution (28 by 28 pixels). We will use 60,000 images 
# to train the network and 10,000 images to evaluate how accurately the network learned to classify images.
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


# Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Set image labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data 
train_images = train_images / 255.0
test_images = test_images / 255.0


# ==============================================================================
# Build the model
# https://keras.io/models/about-keras-models/
# https://keras.io/getting-started/sequential-model-guide/
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# (Alternative 1) You can also simply add layers via the .add() method:
#model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(28, 28)))
#model.add(keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# Print a summary representation of your model
model.summary()

# ==============================================================================


# ==============================================================================
# Compile the model
# https://keras.io/optimizers/ 
# https://keras.io/losses/
# https://keras.io/metrics/
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# ==============================================================================


# ==============================================================================
# serialize model to YAML and save
model_yaml = model.to_yaml()
dir = 'tools/02_basic_classification_ext/'
with open(dir + 'model_basic_classification.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)

# serialize weights to HDF5
model.save_weights(dir + 'model.h5')
print("Saved model to disk")
# ==============================================================================


# Make predictions

# an image
i = 0
img = test_images[i]
img = (np.expand_dims(img,0))
predictions_array = model.predict(img)
result = np.argmax(predictions_array[0])
predicted_label = class_names[result]
label = class_names[test_labels[i]]
print("Image {}: {} {:2.0f}% ({})".format(i, predicted_label, np.max(predictions_array)*100, label))

# images
for i in range(25):
    img = test_images[i]
    img = (np.expand_dims(img,0))
    predictions_array = model.predict(img)
    result = np.argmax(predictions_array[0])
    predicted_label = class_names[result]
    label = class_names[test_labels[i]]
    print("Image {}: {} {:2.0f}% ({})".format(i, predicted_label, np.max(predictions_array)*100, label))

