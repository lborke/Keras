
# cd D:\python\tf_keras

### clean run
# tested on Python 3.7.3 64 bit

# import os

# [opt] disable warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

# Print a summary representation of your model
model.summary()


for layer in model.layers[:87]:
    layer.trainable=False


for layer in model.layers[87:]:
    layer.trainable=True


model.summary()


# local path: klein
# train_dir = './train/'
# paperspace path
# train_dir = '/storage/train/'

# local path: BigSetFull
train_dir = 'T:/temp_data/alltours/train'
validation_dir = 'T:/temp_data/alltours/validation'
test_dir = 'T:/temp_data/alltours/test'


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)


validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')


model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['acc'])


step_size_train = train_generator.n//train_generator.batch_size
step_size_train


history = model.fit_generator(
               train_generator,
               steps_per_epoch = step_size_train,
               epochs = 1,
               # epochs = 30)
               # epochs = 100)
               validation_data=validation_generator,
               validation_steps=47,
               verbose=1)


# ca. 188 sek pro Epoche auf Ryzen
# Epoch 1/1 93/93 - 188s 2s/step - loss: 0.5926 - acc: 0.7618 - val_loss: 0.6895 - val_acc: 0.7160


# We can now finally evaluate this model on the test data:

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')


test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

test_loss,
test_acc


