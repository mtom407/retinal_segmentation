import os
import cv2
import sys
import numpy as np 
from retina_seg.ip_pack import MT_method_processing
#from concurrent.futures import ThreadPoolExecutor


# data surce
DIR = r'data'

# create results dir
MT_dir = r'results'
if not os.path.isdir(MT_dir):
    os.mkdir(MT_dir)

for MASTER_DIR in [MT_dir]:
    vessel_seg_path = os.path.join(MASTER_DIR, 'vessel_seg_processing')
    if not os.path.isdir(vessel_seg_path):
        os.mkdir(vessel_seg_path)

# helper function for segmenting with IP
def MT_processing_seg_helper(img_path):
    DESTINATION = os.path.join(MT_dir, 'vessel_seg_processing')

    im = cv2.imread(img_path)
    segmented_img = MT_method_processing(im)
    new_name = "segmented" + img_path.split('\\')[-1]
    save_path = os.path.join(DESTINATION, new_name)
    cv2.imwrite(save_path, segmented_img)
    print('Saved image to: ' + save_path)

# files to segment
file_paths = [os.path.join(DIR, fname) for fname in os.listdir(DIR)]

# do the thing
for file_path in file_paths:
    MT_processing_seg_helper(file_path)