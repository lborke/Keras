
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


## Load
# Load YAML and create model
# with open('model.yaml', 'r') as yaml_file: loaded_model_yaml = yaml_file.read()
with open('/storage/model.yaml', 'r') as yaml_file: loaded_model_yaml = yaml_file.read()

model = model_from_yaml(loaded_model_yaml)
# load weights into new model
# model.load_weights('model.h5')
model.load_weights('/storage/model.h5')

# Print a summary representation of your model
model.summary()


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
# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())

# oder als Dict definieren (gleicher Task)
labels = {0: 'Detailbilder', 1: 'Hauptbilder', 2: 'Zimmerbilder'}

predictions = [labels[k] for k in predicted_class_indices]


filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

results


# save to csv
results.to_csv("results.csv",index=False)






### check

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

