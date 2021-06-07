import numpy as np
import matplotlib.pyplot as plt
import cv2

def __getCircle(radius):
    x = np.arange(-radius, radius)
    y = np.arange(-radius, radius)
    xx, yy = np.meshgrid(x, y)

    return np.sqrt(xx**2 + yy**2) < radius

def __getGaussian(radius):
    x = np.linspace(-radius, +radius, radius*2)
    y = np.linspace(-radius, +radius, radius*2)
    xx, yy = np.meshgrid(x, y)

    d = np.sqrt(xx**2+yy**2)
    sigma, mu = radius/2, 0.0
    return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

def centroids2Images(point_list, im_num_row, im_num_col, g_radius=20):

    circle_mat = __getGaussian(g_radius)

    temp_im = np.zeros((im_num_row+g_radius*2, im_num_col+g_radius*2))

    for one_pnt in point_list:
        pnt_row = one_pnt[0]
        pnt_col = one_pnt[1]

        temp_im[g_radius+pnt_row-g_radius:g_radius+pnt_row+g_radius, g_radius+pnt_col-g_radius:g_radius+pnt_col+g_radius] = circle_mat

    temp_im = temp_im[g_radius:-g_radius, g_radius:-g_radius]

    return temp_im
