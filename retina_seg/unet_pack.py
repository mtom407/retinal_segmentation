########################################################################################
#                      Scipt author: Michał Tomaszewski                                #
#                   Implementation based on ideas presented in:                        #
# Retina Blood Vessel Segmentation Using A U-Net BasedConvolutional Neural Network     #
# by Xiancheng, W., Wei, L., Bingyi, M., He, J., Jiang, Z., Xu, W., ... & Zhaomeng, S. #
# Folowing functions:                                                                  #
#     equalize_set, adjust_gamma                                                       #
# were borrowed from: https://github.com/orobix/retina-unet            #
########################################################################################

import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
from PIL import Image
import configparser
import h5py

from tensorflow.keras.layers import Input, Conv2D, Dropout, UpSampling2D, MaxPooling2D, Permute, Reshape, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model


def standarize_set(imgs, wtf = False, from_file = False, filename = 'standarization_params.txt'):
    '''Perform standarization on a batch of images based on values taken from a text file'''

    assert wtf != from_file, 'Simultaneous writing and reading from file not supported!'
    
    if from_file:
        # read params from file and make variables
        from_file_params_dict = get_standarization_params(filename)
        global_mean = from_file_params_dict['mean'] 
        global_std = from_file_params_dict['std'] 
    else:
        # calculate global parameters
        global_std = np.std(imgs)
        global_mean = np.mean(imgs)

    # normalize
    normalized_set = (imgs - global_mean) / global_std

    # range to [0, 1]
    for image_id, image in enumerate(normalized_set):
        normalized_set[image_id] = (image - np.min(image))/(np.max(image) - np.min(image)) 
        
    # write to file    
    if wtf:
        with open(filename, 'w') as f:
            print('[params]', end = '\n', file = f)
            print('std = {}'.format(global_std), end = '\n',  file = f)
            print('mean = {}'.format(global_mean), end = '\n',  file = f)
        
    return normalized_set


def equalize_set(imgs):
    '''Perform CLAHE operation on a batch of images'''
    # placeholder
    equalized_set = np.zeros_like(imgs)
    
    # CLAHE object
    clahe = cv2.createCLAHE(clipLimit = 8.0, tileGridSize = (15,15))
    
    # loop over and equalize
    for image_id, image in enumerate(imgs):
        equalized_set[image_id] = clahe.apply(np.array(image*255, dtype = 'uint8'))
        
    return equalized_set


def adjust_gamma(imgs, gamma=1.0):
    '''Perform gamma adjustment on a batch of images'''
    # placholder 
    adjusted_set = np.zeros_like(imgs)
    
    # inverted Gamma
    invGamma = 1.0/gamma
    
    # ============================================================================ #
    # Gamma values < 1 will shift the image towards the darker end of the spectrum #
    #                                                                              #
    # Gamma values > 1 will make the image appear lighter                          #
    # ============================================================================ #
    
    # build lookup table
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    for image_id, image in enumerate(imgs):
        # apply gamma correction using the lookup table
        adjusted_set[image_id] = cv2.LUT(np.array(image, dtype = 'uint8'), table)
    
    return adjusted_set


def onehot_encode_masks(masks):
    '''Turn a binary mask into an encoded mask tensor'''
    # prepare placeholder
    num_masks, mask_height, mask_width, _ = masks.shape
    encoded_masks = np.zeros((num_masks, mask_height, mask_width, 2))
    
    # loop over all mask_patches
    for patch_id, patch in enumerate(masks):
        # unroll mask to vector
        patch = patch.reshape(mask_height*mask_width)
        # placeholder for this mask
        new_patch = np.zeros((mask_height*mask_width, 2))
        # loop over pixels in patch
        for pixel_id, pixel in enumerate(patch):
            # do ya thing baby gurl
            if pixel == 0:
                new_patch[pixel_id, 0] = 1
                new_patch[pixel_id, 1] = 0
            else:
                new_patch[pixel_id, 0] = 0
                new_patch[pixel_id, 1] = 1
        # 2D --> 3D
        new_patch = new_patch.reshape(mask_height, mask_width, 2)
        # save
        encoded_masks[patch_id] = new_patch
        
    return encoded_masks


def get_standarization_params(filename):
    '''Read standarization parameters (during training phase) from a file'''
    
    param_dict = dict()
    
    parser = configparser.ConfigParser()
    parser.read(filename)
    
    param_dict['mean'] = float(parser['params']['mean'])
    param_dict['std'] = float(parser['params']['std'])
    
    return param_dict


def pad_with_zeros(img, *desired_shape):
    '''Pad an image with a border made up of 0's'''
    desired_height = desired_shape[0]
    desired_width = desired_shape[1]
    
    # calculate borders to pad with
    border_x = (desired_width - img.shape[1])//2
    border_y = (desired_height - img.shape[0])//2
    
    # check dimensions
    num_of_dims = len(img.shape)
    if num_of_dims == 2:
        # setup returned array accordingly
        desired_array = np.zeros((desired_height, desired_width), dtype = 'uint8')
        # check for odd border values and correct
        if border_y % 2 != 0:
            border_y += 1 
        elif border_x % 2 != 0:
            border_x += 1
        else:
            pass
        
        # setup borders
        upper_border = np.zeros((border_y, img.shape[1]), dtype = 'uint8')
        lower_border = upper_border.copy() 
        left_border = np.zeros((desired_height, border_x), dtype = 'uint8')
        right_border = left_border.copy()

        # add upper border
        padded_img = np.vstack((upper_border, img))
        # add lower border
        padded_img = np.vstack((padded_img, lower_border))
        # add left border
        padded_img = np.hstack((left_border, padded_img))
        # add right border
        padded_img = np.hstack((padded_img, right_border))

        return padded_img
        
    elif num_of_dims == 3:
        # setup returned array accordingly
        num_channels = img.shape[-1]
        desired_array = np.zeros((desired_height, desired_width, num_channels), dtype = 'uint8')
        # check for odd border values and correct
        if border_y % 2 != 0:
            border_y += 1 
            desired_height += 1
            desired_array = np.zeros((desired_height, desired_width, num_channels), dtype = 'uint8')
        elif border_x % 2 != 0:
            border_x += 1
            desired_width += 1
            desired_array = np.zeros((desired_height, desired_width, num_channels), dtype = 'uint8')
        else:
            pass
        # iterate through channels
        for channel_id in range(num_channels):
            # get channel
            channel_array = img[:,:,channel_id].copy()
            
            # setup borders
            upper_border = np.zeros((border_y, channel_array.shape[1]), dtype = 'uint8')
            lower_border = upper_border.copy()
            left_border = np.zeros((desired_height, border_x), dtype = 'uint8')
            right_border = left_border.copy()

            # add borders
            channel_array = np.vstack((upper_border, channel_array)) 
            channel_array = np.vstack((channel_array, lower_border))
            channel_array = np.hstack((left_border, channel_array))
            channel_array = np.hstack((channel_array, right_border))
            
            desired_array[:,:,channel_id] = channel_array
            
        return desired_array 
    else:
        raise Exception("Object passed in for 'img' argument can be at most 3-dimensional.")


def load_hdf5(infile):
    with h5py.File(infile,"r") as f:
        return f["image"][()]


def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_patch(img,x,y,cropx,cropy):
    # according to what point should the image be cropped?
    startx = x
    starty = y  
    
    return img[starty-cropy:starty+cropy,startx-cropx:startx+cropx]


def get_unet(n_ch,patch_height,patch_width):
    '''Assemble UNet architecture'''
    # down section 1
    inputs = Input(shape=(patch_height,patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # down section 2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # down section 3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv3)
    # up section 1
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv4)
    # up section 2
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv5)
    # up section 3
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same', data_format='channels_last')(conv5)
    conv6 = Reshape((2,patch_height*patch_width))(conv6)
    conv6 = Permute((2,1))(conv6)
    # classification layer
    conv7 = Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

# Following functions taken from F. CHollet - Deep Learning with Python

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def visualize_training(history_dict, smoothed = False, smooth_factor = 0.8):
    
    acc = history_dict['acc']
    loss = history_dict['loss']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    
    if smoothed:
        matplotlib.pyplot.figure(figsize = (12,12))
        matplotlib.pyplot.subplot(221)
        matplotlib.pyplot.plot(epochs, smooth_curve(loss, smooth_factor), 'bo', label = 'Strata trenowania')
        matplotlib.pyplot.plot(epochs, smooth_curve(val_loss, smooth_factor), 'b', label = 'Strata walidacji')
        matplotlib.pyplot.title('Strata trenowania i walidacji')
        matplotlib.pyplot.xlabel('Epoki')
        matplotlib.pyplot.ylabel('Strata')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.subplot(222)
        matplotlib.pyplot.plot(epochs, smooth_curve(acc, smooth_factor), 'bo', label = 'Dokładność trenowania')
        matplotlib.pyplot.plot(epochs, smooth_curve(val_acc, smooth_factor), 'b', label = 'Dokładnośc walidacji')
        matplotlib.pyplot.title('Dokładność trenowania i walidacji')
        matplotlib.pyplot.xlabel('Epoki')
        matplotlib.pyplot.ylabel('Dokładność')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.figure(figsize = (12,12))
        matplotlib.pyplot.subplot(221)
        matplotlib.pyplot.plot(epochs, loss, 'bo', label = 'Strata trenowania')
        matplotlib.pyplot.plot(epochs, val_loss, 'b', label = 'Strata walidacji')
        matplotlib.pyplot.title('Strata trenowania i walidacji')
        matplotlib.pyplot.xlabel('Epoki')
        matplotlib.pyplot.ylabel('Strata')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.subplot(222)
        matplotlib.pyplot.plot(epochs, acc, 'bo', label = 'Dokładność trenowania')
        matplotlib.pyplot.plot(epochs, val_acc, 'b', label = 'Dokładnośc walidacji')
        matplotlib.pyplot.title('Dokładność trenowania i walidacji')
        matplotlib.pyplot.xlabel('Epoki')
        matplotlib.pyplot.ylabel('Dokładność')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.show()