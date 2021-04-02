########################################################################################
#                      Scipt author: Michał Tomaszewski                                #
#                   Implementation based on ideas presented in:                        #
# Retina Blood Vessel Segmentation Using A U-Net BasedConvolutional Neural Network     #
# by Xiancheng, W., Wei, L., Bingyi, M., He, J., Jiang, Z., Xu, W., ... & Zhaomeng, S. #
# Folowing functions:                                                                  #
#     equalize_set, adjust_gamma                                                       #
# and methodology were borrowed from: https://github.com/orobix/retina-unet            #
########################################################################################

import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
from PIL import Image
from retina_seg.unet_pack import standarize_set, equalize_set, adjust_gamma
from retina_seg.unet_pack import get_unet, onehot_encode_masks, get_patch, visualize_training
from retina_seg.glaucoma_functions import visualize_training
from tensorflow.keras.callbacks import ModelCheckpoint

TRAINING_DIR = r''
MASKS_DIR = r''


num_images = len(os.listdir(TRAINING_DIR))
im_height = 584
im_width = 565
im_depth = 3

#all_images = np.zeros((num_images, im_height, im_width, im_depth))
#all_greens = np.zeros_like(all_images[:,:,:,0])
all_greens = np.zeros((num_images, im_height, im_width))
all_masks = np.zeros_like(all_greens)

for fid, (fname_img, fname_mask) in enumerate(zip(os.listdir(TRAINING_DIR), os.listdir(MASKS_DIR))):
    # reading path
    read_path = os.path.join(TRAINING_DIR, fname_img)
    read_path_mask = os.path.join(MASKS_DIR, fname_mask)
    
    # read image
    image = cv2.cvtColor(cv2.imread(read_path), cv2.COLOR_BGR2RGB)
    image_green = image[:,:,1]
    
    # read mask
    mask = np.array(Image.open(read_path_mask), dtype = 'uint8')
    mask[mask == 255] = 1
    
    # save image
    #all_images[fid, :, :, :] = image/255
    all_greens[fid, :, :] = image_green/255
    # save mask
    all_masks[fid,:,:] = mask


# preprocessing
adjusted_dataset_green = standarize_set(all_greens, wtf = True, filename = 'green_set_training.txt')
adjusted_dataset_green = equalize_set(adjusted_dataset_green)
adjusted_dataset_green = adjust_gamma(adjusted_dataset_green, 1.1)

# EXTRACTING PATCHES
sample_image = adjusted_dataset_green[0]
sample_mask = all_masks[0]

im_height, im_width = sample_image.shape
patch_height, patch_width = 32, 32

# acceptable coordinates for patch_centers
start_x = 0 + patch_width
end_x = im_width - patch_width

start_y = 0 + patch_height
end_y = im_height - patch_height

print('Image dimensions: ({}, {})'.format(im_height, im_width))

num_patches = 9500
patches = np.zeros((num_patches*num_images, patch_height, patch_width, 1))
mask_patches = np.zeros_like(patches)

# DATA AUGMENTATION - sampling 9500 patches from every training image + mask pair
for pair_id, (image, mask) in enumerate(zip(adjusted_dataset_green, all_masks)):
    for patch_id in range(num_patches):

        patch_center_x = np.random.randint(low = start_x, high = end_x)
        patch_center_y = np.random.randint(low = start_y, high = end_y)

        patch = get_patch(image, patch_center_x, patch_center_y, patch_width//2, patch_height//2)
        mask_patch = get_patch(mask, patch_center_x, patch_center_y, patch_width//2, patch_height//2)

        patches[pair_id*num_patches+patch_id, :, :, 0] = patch
        mask_patches[pair_id*num_patches+patch_id, :, :, 0] = mask_patch


del adjusted_dataset_green
del all_greens

# ENCODING MASKS
encoded_mask_patches = onehot_encode_masks(mask_patches)
print(encoded_mask_patches.shape)

# TRAIN/VAL split and counting samples
dataset_sample_count = len(patches)
train_val_split = int(dataset_sample_count*0.9)

train_set = patches[:train_val_split]
train_masks = encoded_mask_patches[:train_val_split]
val_set = patches[train_val_split:]
val_masks = encoded_mask_patches[train_val_split:]

print('Train samples: {}'.format(len(train_set)))
print('Validation samples: {}'.format(len(val_set)))

del mask_patches
del encoded_mask_patches
del patches

# MODEL setup
unet_model = get_unet(1, patch_height, patch_width)
unet_model.summary()
# callback setup 
my_callback = ModelCheckpoint(filepath='test_demo_chkpt.hdf5',monitor = 'val_loss',  verbose=1, save_best_only=True)

# MODEL input prep
flattened_masks_train = train_masks.reshape(len(train_set), patch_height*patch_width, 2)
flattened_masks_val = val_masks.reshape(len(val_set), patch_height*patch_width, 2)

# MODEL training
history = unet_model.fit(train_set, flattened_masks_train,
              batch_size = 16,
              epochs = 2,
              callbacks = [my_callback],
              validation_data = (val_set, flattened_masks_val))

visualize_training(history.history)