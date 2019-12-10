
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


# tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel('WARNING')


# geht !!
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 0     | DEBUG            | [Default] Print all messages       
# 1     | INFO             | Filter out INFO messages           
# 2     | WARNING          | Filter out INFO & WARNING messages 
# 3     | ERROR            | Filter out all messages 


## test

import keras
keras.__version__


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


