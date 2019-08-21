
# cd D:\python\tf_keras

### clean run
# tested on Python 3.7.3 64 bit

import os
import keras

# import pillow
from PIL import Image

from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
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


# check
for i,layer in enumerate(model.layers):
  print(i,layer.name)


for i,layer in enumerate(model.layers[:87]):
  print(i,layer.name)


for i,layer in enumerate(model.layers[87:]):
  print(i,layer.name)

# ENDE check


for layer in model.layers[:87]:
    layer.trainable=False


for layer in model.layers[87:]:
    layer.trainable=True


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

# train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
train_generator=train_datagen.flow_from_directory('D:/TEMP/alltours/BigSetFull', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)



model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs = 10)


