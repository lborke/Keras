
# cd D:\python\tf_keras

### clean run
# tested on Python 3.7.3 64 bit

import os
import keras

# import pillow
from PIL import Image

# for prediction
import numpy as np
import pandas as pd

from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
# from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, model_from_yaml
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



## Test the model
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    directory='./test/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False
)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
                            steps=STEP_SIZE_TEST,
                            verbose=1)


predicted_class_indices=np.argmax(pred,axis=1)

# train_datagen wird benötigt
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

# oder als Dict definieren (gleicher Task)
labels = {0: 'Detailbilder', 1: 'Hauptbilder', 2: 'Zimmerbilder'}

predictions = [labels[k] for k in predicted_class_indices]


filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

# save to csv
results.to_csv("results.csv",index=False)



## Save
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# serialize model to YAML
model_yaml = model.to_yaml()
with open('model.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)


# serialize weights to HDF5
model.save_weights("model.h5")
# print("Saved model to disk")


## Load
# Load YAML and create model
with open('model.yaml', 'r') as yaml_file: loaded_model_yaml = yaml_file.read()

model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights('model.h5')

# Print a summary representation of your model
model.summary()




### check

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)


predictions_array = model.predict(img)



### Stats
## alle ~ 10.000 Bilder
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


