
## Docker runs
# tensorflow:1.14.0-gpu-py3 / OpenCV / keras_segmentation
# P4000
sudo docker start -ai 4a8add41ba16


#
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1

cv2.imwrite("ann_1.png", ann_img)


#

from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12

model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="input_image.jpg",
    out_fname="out.png"
)


#

out = model.predict_segmentation(
    inp = "sample_predict/bild1.jpg",
    out_fname = "out/bild1.png"
)


out = model.predict_segmentation(
    inp = "sample_predict/bild2.jpg",
    out_fname = "out/bild2.png"
)


out = model.predict_segmentation(
    inp = "sample_predict/bild3.jpg",
    out_fname = "out/bild3.png"
)


out = model.predict_segmentation(
    inp = "sample_predict/bild4.jpg",
    out_fname = "out/bild4.png"
)


out = model.predict_segmentation(
    inp = "sample_predict/bild5.jpg",
    out_fname = "out/bild5.png"
)


out = model.predict_segmentation(
    inp = "sample_predict/bild6.jpg",
    out_fname = "out/bild6.png"
)


##

# /data/sample_predict
# /data/out

