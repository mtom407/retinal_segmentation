# retinal_segmentation
Retinal vessel segmentation with deep learning UNet architecture and image processing methods

Part of the final year engineering project from my studies at Warsaw's University of Technology.

**Disclaimer**: The training scripts presented in this repository require medical data in order to train and test deep learning models. I myself cannot share this data here. You can check out the [DRIVE](https://drive.grand-challenge.org/) dataset yourself - this data was used to train and initially test the UNet architecture. The best model was then tested on data aquired from an ophtalmic clinic located in Warsaw - this data will not be shared here either but you can see the results and how you can test the algorithms by yourself below.  

This project aims to segment retinal vessels from funduscamera images. Two techniques were used to achieve this goal:
- image processing techqniues presented in: [An Efficient Retinal Blood Vessel Segmentation using Morphological Operations](https://www.researchgate.net/publication/329565456_An_Efficient_Retinal_Blood_Vessel_Segmentation_using_Morphological_Operations)
- personal implementation of a deep learning techique using the fully convolutional UNet as presented in: [Retina Blood Vessel Segmentation Using A U-Net BasedConvolutional Neural Network](https://researchbank.swinburne.edu.au/file/fce08160-bebd-44ff-b445-6f3d84089ab2/1/2018-xianchneng-retina_blood_vessel.pdf)

Example of the results: Original | Image Processing | UNet

![Segmentation showcase](https://github.com/mtom407/retinal_segmentation/blob/main/docs/images/showcase_1.png)

If you'd like to test the UNet for yourself you might want to:
1. Get a funduscamera image (it must be cropped to the FOV cricle first) like: [example image](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Fundus_photograph_of_normal_right_eye.jpg/1200px-Fundus_photograph_of_normal_right_eye.jpg)
2. Create a 'data' folder in the repository and save the image there as 'example_1.png'
3. Run unet_showcase.py

