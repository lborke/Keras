
## Docker runs
# tensorflow:1.14.0-gpu-py3 / OpenCV / keras_segmentation
# P4000
sudo docker start -ai 4a8add41ba16



##

# /data/sample_predict
# /data/out


from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12, resnet_pspnet_VOC12_v0_1

from keras_segmentation.predict import predict_multiple


# TF 2.0 error: module 'tensorflow_core._api.v2.image' has no attribute 'resize_images'
predict_arr = predict_multiple( 
	model = pspnet_101_voc12(), 
	inp_dir = "/data/sample_predict/", 
	out_dir = "/data/out/" 
)


# temp
predict_arr = predict_multiple( 
	model = pspnet_101_voc12(), 
	inp_dir = "/data/sample2/", 
	out_dir = "/data/out/" 
)



# TF 2.0 geht auch
predict_arr = predict_multiple( 
	model = resnet_pspnet_VOC12_v0_1(), 
	inp_dir = "/data/sample_predict/", 
	out_dir = "/data/out/" 
)


# TF 2.0 error: module 'tensorflow_core._api.v2.image' has no attribute 'resize_images'
predict_arr = predict_multiple( 
	model = pspnet_50_ADE_20K(), 
	inp_dir = "/data/sample_predict/", 
	out_dir = "/data/out/" 
)


# TF 2.0 error: module 'tensorflow_core._api.v2.image' has no attribute 'resize_images'
predict_arr = predict_multiple( 
	model = pspnet_101_cityscapes(), 
	inp_dir = "/data/sample_predict/", 
	out_dir = "/data/out/" 
)






len(predict_arr)

predict_arr[0].shape


##
# https://unix.stackexchange.com/questions/19654/how-do-i-change-the-extension-of-multiple-files

# Rename all *.txt to *.text
for f in *.JPG; do 
    mv -- "$f" "${f%.JPG}.jpg"
done



