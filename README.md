# Centroid Unet - Locating Object Centroids in Satellite Images

### Introduction

Centroid-UNet is deep neural network model to detect centroids from satellite images. We have tested this model with two case studies (building centroid detection and agriculture plot centroid detection). This network is based on classic U-Net sematic segmentation architecture. Satellite RGB images are used as input data for the model. And images generated with Gaussians surrounding centroids of instances are used as target for the model. Sample input and prediction from the model is shown in below figure.

### How to Use

Input data are RGB satellite images. Target data can be given as JSON files with list of centroids points in each satellite image tiles. First Gaussians are generated around lists of centroids points, then images are generated as targets for the model during preprocessing steps. If we use centroids as it is without Gaussians, training process will be challenging. Using Gaussians instead of just centroids make training process more stable. Sample data is also given with this repository. 

Our model is U-Net architecture which was written in Keras with Tensorflow backend. You can modify the model according to your requirement as well.

### Libraries used
- numpy
- matplotlib
- imageio
- keras (tensorflow)

### References







