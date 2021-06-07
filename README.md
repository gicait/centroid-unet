# Centroid Unet

### Introduction

Centroid-UNet is deep neural network model to detect centroids from satellite images. We have tested this model with the building centroid detection and agriculture plot centroid detection case studies. This network is based on classic U-Net sematic segmentation architecture. Satellite RGB images are used as input data for the model. And images generated with Gaussians surrounding centroids of instances are used as target for the model. Sample input and prediction from the model is shown in below figure.

### How to Use

Input data are RGB satellite images. Target data can be given as JSON files with list of centroids points in each satellite image tiles. Sample data is also given with this repository. Surrounding lists of centroids points, Gaussians are generated and images are generated during preprocessing steps.

Model is simple U-Net architecture which was written in Keras with Tensorflow backend. You can modify the model according to your requirement as well.

### Libraries used
- numpy
- matplotlib
- tensorflow v1.12.0
- keras
- sklearn

### References






