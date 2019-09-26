
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



##
Epoch 1/10
299/299 [==============================] - 414s 1s/step - loss: 0.4908 - acc: 0.7925
Epoch 2/10
299/299 [==============================] - 405s 1s/step - loss: 0.3991 - acc: 0.8270
Epoch 3/10
299/299 [==============================] - 408s 1s/step - loss: 0.3738 - acc: 0.8401
Epoch 4/10
299/299 [==============================] - 408s 1s/step - loss: 0.3643 - acc: 0.8431
Epoch 5/10
299/299 [==============================] - 406s 1s/step - loss: 0.3357 - acc: 0.8522
Epoch 6/10
299/299 [==============================] - 405s 1s/step - loss: 0.3287 - acc: 0.8544
Epoch 7/10
299/299 [==============================] - 405s 1s/step - loss: 0.3029 - acc: 0.8647
Epoch 8/10
299/299 [==============================] - 408s 1s/step - loss: 0.2954 - acc: 0.8699
Epoch 9/10
299/299 [==============================] - 406s 1s/step - loss: 0.2715 - acc: 0.8791
Epoch 10/10
299/299 [==============================] - 407s 1s/step - loss: 0.2594 - acc: 0.8844
<keras.callbacks.History object at 0x0000020ED6205780>


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


