########################################################################################
#                           Script author: Michał Tomaszewski                          #
#            MT_method_processing function implementation is based on:                 #
#     An Efficient Retinal Blood Vessel Segmentation using Morphological Operations    #
#                  Ozkava, U., Ozturk, S., Akdemir, B., & Sevfi, L.                    #
########################################################################################

import numpy as np
import cv2
from scipy.signal import wiener as WienerFilter


def rgb2xyz(image):
    '''Convert a RGB image into a XYZ one'''
    im_height, im_width, im_depth = image.shape
    image = image.reshape((im_height*im_width, im_depth))

    # conversion array
    A = np.array([[0.4184, -0.1586, -0.0828],
                  [-0.0911, 0.2524, 0.0157],
                  [0.0009, -0.0025, 0.1786]])

    # calc new values for each pixel
    new_img = np.zeros_like(image)
    for pix_id, pixel in enumerate(image):
        xyz_vec = np.linalg.solve(A, pixel)
        new_img[pix_id, :] = xyz_vec

    new_img = new_img.reshape((im_height, im_width, im_depth))

    return new_img


def MT_method_processing(image_array):
    '''Perform retinal vessel segmentation with image processing techniques'''
    # crop to main circle first !!!
    # the following values are problem specific
    # starty = 1536
    # startx = 2048
    # cropy = 1393
    # cropx = 1393
    # image_array = image_array[starty-cropy:starty+cropy,startx-cropx:startx+cropx]

    image_array = cv2.resize(image_array, (565, 584))
    # greeen_channel
    green_channel = image_array[:, :, 1].copy()

    # equalize
    clahe = cv2.createCLAHE(4.0, (8, 8))
    green_channel = clahe.apply(green_channel)

    # adaptive thresholding with gaussian windows
    thresholded = cv2.adaptiveThreshold(green_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 13)

    # sharpening - converting RGB2XYZ
    thresh_rgb = np.zeros_like(image_array)
    thresh_rgb[:, :, 1] = thresholded
    xyz_image = rgb2xyz(thresh_rgb)

    # sharpening - setting up image in L*a*b
    L_band = 100 * np.sqrt(xyz_image[:, :, 1])
    a_band = 172.3 * ((xyz_image[:, :, 0] - xyz_image[:, :, 1])/L_band)
    b_band = 67.2 * ((xyz_image[:, :, 1] - xyz_image[:, :, 2])/L_band)
    lab_img = np.zeros_like(xyz_image)
    lab_img[:, :, 0] = L_band
    lab_img[:, :, 1] = a_band
    lab_img[:, :, 2] = b_band

    # sharpening - kernel setup
    sharpening_filter = -1*np.ones((3, 3))
    sharpening_filter[1, 1] = 9

    # sharpeninig - sharpening the L band and converting L*a*b --> RGB
    sharpened_L_band = cv2.filter2D(lab_img[:, :, 0], -1, sharpening_filter)
    lab_img[:, :, 0] = np.float32(sharpened_L_band)
    lab2rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    lab2rgb_green_channel = lab2rgb_img[:, :, 1]
    ret, sharpened = cv2.threshold(lab2rgb_green_channel, 128, 255, cv2.THRESH_BINARY)

    # denoising
    denoised_green = WienerFilter(
        lab2rgb_green_channel, mysize=3, noise=1.5).astype('uint8')
    # Otsu
    ret2, otsu_img = cv2.threshold(
        denoised_green, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # otwarcie morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(otsu_img, cv2.MORPH_OPEN, kernel)

    return opening
