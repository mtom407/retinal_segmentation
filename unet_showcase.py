########################################################################################
#                      Scipt author: Michał Tomaszewski                                #
#                   Implementation based on ideas presented in:                        #
# Retina Blood Vessel Segmentation Using A U-Net BasedConvolutional Neural Network     #
# by Xiancheng, W., Wei, L., Bingyi, M., He, J., Jiang, Z., Xu, W., ... & Zhaomeng, S. #
# Folowing functions:                                                                  #
#     equalize_set, adjust_gamma                                                       #
# and methodology were borrowed from: https://github.com/orobix/retina-unet            #
########################################################################################

from tensorflow.keras.models import load_model
import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt

from retina_seg.unet_processors import DataProcessor, VesselExtractor


TEST_DIR = r'data'
dp = DataProcessor(TEST_DIR)

# results directory
MT_dir = r'results'
if not os.path.isdir(MT_dir):
    os.mkdir(MT_dir)
    save_dir = os.path.join(MT_dir, 'vessel_seg_neural_net')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
else:
    save_dir = os.path.join(MT_dir, 'vessel_seg_neural_net')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

# number of images and their dimensions
num_images = 1
im_height = 544
im_width = 544

test_images = dp.load(num_images, im_height, im_width)
adjusted_images= dp.preprocess(from_file=True, filename='green_set_training.txt')

# check the dimensions once again (may be affected by pad_with_zeros)
sample_image = adjusted_images[0]
im_height, im_width = sample_image.shape
print(f'Image height = {im_height}, image width =  {im_width}')

# slice & prediction method parameters (patch dimensions should be the same as training)
stride = 2
patch_height, patch_width = 32, 32

# load the model and run predictions
loaded_model = load_model('green_model.hdf5')
ve = VesselExtractor(adjusted_images, patch_height, patch_width, stride, loaded_model)
ve.reconstruct_directory(save_dir)

# testing the other function
sliced_sample = ve.slice_image(sample_image)
processed_image = ve.reconstruct_image(sliced_sample)
plt.figure()
plt.imshow(processed_image, 'gray')
plt.show()


