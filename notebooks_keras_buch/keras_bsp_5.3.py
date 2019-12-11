
# disable warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import keras
keras.__version__


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


## simple feature extraction model

import numpy as np

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])


history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


## data augmentation model

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))


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
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=1)


# ca. 150 sek pro Epoche auf Ryzen
# Epoch 2/2 - 149s - loss: 0.4621 - acc: 0.8065 - val_loss: 0.3580 - val_acc: 0.8740
# Epoch 15/15 100/100 - 150s 2s/step - loss: 0.3151 - acc: 0.8625 - val_loss: 0.2466 - val_acc: 0.9010
# (weitere 10 Epochen [15+10]) Epoch 10/10 100/100 - 151s 2s/step - loss: 0.2945 - acc: 0.8720 - val_loss: 0.2365 - val_acc: 0.9070

# model.save('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_15ep.h5')

model.save('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_25ep.h5')



## model Fine-tuning

from keras.models import load_model

# https://stackoverflow.com/questions/49195189/error-loading-the-saved-optimizer-keras-python-raspberry

model = load_model('T:\\temp_data\\cats_and_dogs_small\\cats_and_dogs_small_3_test_25ep.h5')

model.summary()


# conv_base.trainable = True

for layer in model.layers:
    print(layer.name)


model.layers[0].name

model.layers[0].trainable = False


set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False





