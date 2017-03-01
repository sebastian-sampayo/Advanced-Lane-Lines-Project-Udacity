# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 4: Advance Lane Lines
# Date: 26th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: camera_calibration.py
# =========================================================================== #
# Perform camera calibration

import glob
import cv2
import pickle
import matplotlib.pyplot as plt
from utils import *

# --------------------------------------------------------------------------- #
test_img_filename = 'camera_cal/calibration13.jpg'
undistort_example = 'camera_cal/undist_calibration13.jpg'
result_filename = 'camera_cal/test13.jpg'
camera_calibration_filename = 'camera_cal/mtx_dist_pickle.p'

# --------------------------------------------------------------------------- #
# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
nx = 9
ny = 6

# Test undistortion on an image
test_img = cv2.imread(test_img_filename)
img_size = (test_img.shape[1], test_img.shape[0])

# Calibrate
mtx, dist = calibrate_camera(images, nx, ny, img_size, verbose=False)

# Undistort image
dst = cv2.undistort(test_img, mtx, dist, None, mtx)
cv2.imwrite(undistort_example, dst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
ax1.imshow(test_img)
ax1.set_title('Original Image', fontsize=30)
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
# Save result
f.savefig(result_filename)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open(camera_calibration_filename , "wb" ) )