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
from tensorflow.keras.models import model_from_yaml

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
# Load YAML and create model
dir = 'tools/02_basic_classification_ext/'
yaml_file = open(dir + 'model_basic_classification.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights(dir + 'model.h5')
print("Loaded model from disk")

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

# Evaluate accuracy on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
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

