
## opt
# disable warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import keras
keras.__version__


## main
from keras.applications import VGG16

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import os

base_dir = 'T:\\temp_data\\cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


conv_base.summary()


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])


# https://keras.io/models/sequential/
# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      # epochs=30,
      # epochs=15,
      epochs=2,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=1)


# ca. 150 sek pro Epoche auf Ryzen
# Epoch 2/2 - 149s - loss: 0.4621 - acc: 0.8065 - val_loss: 0.3580 - val_acc: 0.8740
# Epoch 15/15 100/100 - 150s 2s/step - loss: 0.3151 - acc: 0.8625 - val_loss: 0.2466 - val_acc: 0.9010
# (weitere 10 Epochen [15+10]) Epoch 10/10 100/100 - 151s 2s/step - loss: 0.2945 - acc: 0.8720 - val_loss: 0.2365 - val_acc: 0.9070

# model.save('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_15ep.h5')

# model.save('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_25ep.h5')


## Model evaluation


from keras.models import load_model

# https://stackoverflow.com/questions/49195189/error-loading-the-saved-optimizer-keras-python-raspberry

model = load_model('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_25ep.h5')

model.summary()



# We can now finally evaluate this model on the test data:

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

test_loss,
test_acc


print('test acc:', test_acc)


