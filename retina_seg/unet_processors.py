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
import re

from retina_seg.unet_pack import standarize_set, equalize_set, adjust_gamma, pad_with_zeros, get_patch


class DataProcessor:
    '''Object that loads images from a directory and preprocesses them for further analysis '''
    
    def __init__(self, data_directory):
        '''Create an instance by passing the path of the directory holding all the images'''
        file_list = [fname.split('.')[0] for fname in os.listdir(data_directory)]
        file_list.sort(key = self._key)
        file_list = [fname + '.png' for fname in file_list]

        self.file_list = file_list
        self.data_directory = data_directory


    def load(self, num_images, im_height, im_width):
        '''Load images from directory passed when constructing DataProcessor instance'''
        self.images = np.zeros((num_images, im_height, im_width))

        for fid, fname_img in enumerate(self.file_list[:num_images]):

            read_path_img = os.path.join(self.data_directory, fname_img)
            # load and resize the image
            test_img = cv2.cvtColor(cv2.imread(read_path_img), cv2.COLOR_BGR2RGB)
            test_img = cv2.resize(test_img, (im_width, im_height))
            # pad with zeros -> will be needed for later
            padded_test_img = pad_with_zeros(test_img, *(im_height, im_width))
            test_green = padded_test_img[:,:,1]

            # save the image
            self.test_images[fid, :, :] = test_green/255

        return self.images


    def preprocess(self, from_file = True, filename = 'green_set_training.txt'):
        '''Standarize, equalize and gamma adjust a set of images'''
        preprocessed_images = standarize_set(self.test_images, from_file = True, filename = 'green_set_training.txt')
        preprocessed_images = equalize_set(preprocessed_images)
        preprocessed_images = adjust_gamma(preprocessed_images, 1.1)

        return preprocessed_images


    def _key(self, item):
        key_pat = re.compile(r"^(\D+)(\d+)$")
        m = key_pat.match(item)
        return m.group(1), int(m.group(2))


class VesselExtractor:
    '''Object that performs retinal vessel segmentation using a fully convolutional nn: UNet'''
    
    def __init__(self, images, patch_height, patch_width, stride, model):
        '''Create an instance by passing:
        - images : numpy array holding all the images
        - patch_height & patch_width : int number signifying the size of patches to be used for prediction
        - stride : step parameter signifying how much the extracted patches will overlap
        - model : trained UNet model for retinal vessel segmentation task'''
        self.images = images
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride = stride
        self.model = model

        self._parameter_setup()


    def _parameter_setup(self):
        '''Calculate parameters needed for further steps: slicing and image reconstruction'''
        im_height, im_width = self.images[0].shape

        #                  start_x                  end_x
        self.x = np.array([0 + self.patch_width//2, im_width - self.patch_width//2])
        #                  start_y                   end_y
        self.y = np.array([0 + self.patch_height//2, im_height - self.patch_height//2])

        # prediction method parameters
        self.output_depth = (self.patch_height//self.stride) + 1
        self.independent_patches_per_row = im_width // self.patch_width
        self.overall_patches_per_row = len(np.array(range(self.x[0], self.x[1]+1, self.stride)))
        self.independent_rows_per_image = im_height // self.patch_height


    def slice_images(self):
        '''Slice loaded images and return them in a list'''
        images_sliced = []
        for sample_id, sample_image in enumerate(self.images):
            sliced_image = self.slice_image(sample_image)
            images_sliced.append(sliced_image)

        return images_sliced


    def slice_image(self, image):
        '''Slice an image by creating multiple overlapping rows with each row containing overlapping patches for prediction averaging'''
        # track number of rows per image
        rows_per_image_counter = 0
        image_rows = []
        # for every possible y coordinate generate list of overlapping rows
        for patch_center_y in range(self.y[0], self.y[1]+1, self.stride): 
            
            patch_center_id = 0
            # track tensor depth when saving patches
            depth_counter = 0

            # placeholder for all patches generated from just one row of the image
            overall_row_patches = np.zeros((self.overall_patches_per_row, self.patch_height, self.patch_width, 1))

            # for every row generate overlapping patches
            for overall_patch_id, patch_center_x in enumerate(range(self.x[0], self.x[1]+1, self.stride)):

                # check if the depth counter "overflows"
                if depth_counter == self.output_depth:
                    depth_counter = 0
                    patch_center_id -= (self.output_depth-1)

                # generate a patch
                patch = get_patch(image, patch_center_x, patch_center_y, 
                                    self.patch_width//2, self.patch_height//2)
                patch_center = np.array([patch_center_x, patch_center_y])

                # calculate where to put it
                if depth_counter == 0:
                    patch_start_point = patch_center_id*self.patch_width
                    patch_end_point = (patch_center_id+1)*self.patch_width
                else:
                    patch_start_point += self.stride
                    patch_end_point += self.stride

                # save the patch
                overall_row_patches[overall_patch_id, :, :, 0] = patch
                
                # update the counters
                depth_counter += 1
                patch_center_id += 1

            # update counters and append the entire row
            rows_per_image_counter += 1
            image_rows.append(overall_row_patches)

        return image_rows


    def reconstruct_row(self, row):
        '''Run prediction on a row of overlapping patches contained in it and then average the results'''
        # run prediction one one row
        row_prediction = self.model.predict(row)
        # change predictions shape
        row_prediction = row_prediction.reshape((row_prediction.shape[0], self.patch_height, self.patch_width, 2))

        depth_counter = 0
        patch_id = 0
        # this holds prediction results (binary) which will be later averaged
        row2average_vessels = np.zeros((self.patch_height, self.patch_width*self.independent_patches_per_row, self.output_depth))
        # here we are averaging by column
        for patch_prediction in row_prediction:

            if depth_counter == self.output_depth:
                depth_counter = 0
                patch_id -= (self.output_depth-1)

            # calculate where to save newly generated patches
            if depth_counter == 0:
                patch_start_point = patch_id*self.patch_width
                patch_end_point = (patch_id+1)*self.patch_width
            else:
                patch_start_point += self.stride
                patch_end_point += self.stride

            # place the prediction
            vessel_prob = patch_prediction[:,:,1]
            row2average_vessels[:, patch_start_point:patch_end_point, depth_counter] = vessel_prob

            # update
            depth_counter += 1
            patch_id += 1

        return row2average_vessels


    def reconstruct_image(self, sliced_image):
        '''Reconstruct the full image by averaging overlapping rows'''
        # placeholders for averaging operation and counters
        rows2average = np.zeros((self.patch_height*self.independent_rows_per_image, 
                                 self.patch_width*self.independent_patches_per_row, 
                                 self.output_depth))
        row_id = 0
        outer_depth_counter = 0
        for overall_row_id, row in enumerate(sliced_image):

            row_vessels = self.reconstruct_row(row)
            # averaging overlapping columns
            averaged_vessels = np.mean(row_vessels, axis = 2)

            if outer_depth_counter == self.output_depth:
                outer_depth_counter = 0
                row_id -= (self.output_depth-1)

            if outer_depth_counter == 0:
                row_start_point = row_id * self.patch_height
                row_end_point = (row_id+1)*self.patch_height
            else:
                row_start_point += self.stride
                row_end_point += self.stride

            # finally place the averaged prob map in the placeholder
            rows2average[row_start_point:row_end_point, :, outer_depth_counter] = averaged_vessels
            outer_depth_counter += 1
            row_id += 1

        # finally average overlapping rows
        output_prob_map = np.mean(rows2average, axis = 2)
        final_prob_map = output_prob_map[:512, :512]

        return final_prob_map


    def reconstruct_directory(self, save_dir):
        '''Perform patch batch prediction + image reconstruction on every loaded image and save results to a directory'''
        num_images = self.images.shape[0]
        images_sliced = self.slice_images()

        for image_id, sliced_image in enumerate(images_sliced):
            
            final_prob_map = self.reconstruct_image(sliced_image)

            # save it
            save_path = os.path.join(save_dir, f'unet_seg{image_id}.png')
            cv2.imwrite(save_path, final_prob_map*255)