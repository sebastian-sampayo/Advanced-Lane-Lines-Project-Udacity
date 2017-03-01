# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 4: Advance Lane Lines
# Date: 26th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: analyse_thresholds.py
# =========================================================================== #
# Analyse different gradient and color thresholds

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import cv2

from utils import *
# --------------------------------------------------------------------------- #
input_filename = 'test_images/test5.jpg'
camera_calibration_filename = 'camera_cal/mtx_dist_pickle.p'

# --------------------------------------------------------------------------- #
# Load camera calibration matrix and distortion coefficients
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']

# Read in a tests image
image = mpimg.imread(input_filename)

# Undistort test image
image = cv2.undistort(image, mtx, dist, None, mtx)

# --------------------------------------------------------------------------- #
## Gradient Thresholds
# Apply each of the thresholding functions
gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(20, 100))
grady = abs_sobel_threshold(image, orient='y', sobel_kernel=3, thresh=(20, 100))
mag_binary = mag_threshold(image, sobel_kernel=3, thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

## Color Thresholds
S = S_threshold(image, (170, 255))
R = R_threshold(image, (170, 255))
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
S_cont = hls[:,:,2]
S_cont3 = np.dstack((S_cont, S_cont, S_cont))

S_gradx = abs_sobel_threshold(S_cont3, orient='x', sobel_kernel=3, thresh=(20, 100))

# Combine S and S_gradx
S2 = np.zeros_like(S)
S2[ (S==1) & (S_gradx==1) ] = 1

# Combination
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Yellow filter
yellow = yellow_threshold(image)

# White filter
white = white_threshold(image)
   
# --------------------------------------------------------------------------- #
# Binary output
sobel_kernel = 3
thresh = [
         (20, 100), 
         (30, 100), 
         (0.7, 1.3), 
         (170, 255)
         ]
binary = color_and_gradient(image, sobel_kernel=sobel_kernel, 
                            thresh=thresh, verbose=True)

# --------------------------------------------------------------------------- #
## Plots
f, axarr = plt.subplots(3, 2, figsize=(20,18), sharex=True)
axarr[0,0].set_title('Gradient x', fontsize=20)
axarr[0,0].imshow(gradx, cmap='gray')
axarr[0,1].set_title('Gradient y', fontsize=20)
axarr[0,1].imshow(grady, cmap='gray')
axarr[1,0].set_title('Gradient magnitude', fontsize=20)
axarr[1,0].imshow(mag_binary, cmap='gray')
axarr[1,1].set_title('Gradient direction', fontsize=20)
axarr[1,1].imshow(dir_binary, cmap='gray')
axarr[2,0].set_title('Yellow', fontsize=20)
axarr[2,0].imshow(yellow, cmap='gray')
axarr[2,1].set_title('White', fontsize=20)
axarr[2,1].imshow(white, cmap='gray')
f.subplots_adjust(hspace=0.07, wspace=0)
#f.savefig('examples/thresholds.jpg')


## Perspective transform
# src: 4 points in the photo that should be a rectangle in the real world.
# dst: The 4 vertices of a rectangle in the transformed (output) image.
img_size = (image.shape[1], image.shape[0])
src, dst = points_for_warper(img_size)

w_gradx = warper(gradx, src, dst)
w_grady = warper(grady, src, dst)
w_mag = warper(mag_binary, src, dst)
w_dir = warper(dir_binary, src, dst)
w_S = warper(S, src, dst)
w_R = warper(R, src, dst)
w_Scont = warper(S_cont, src, dst)
w_Sgradx = warper(S_gradx, src, dst)
w_S2 = warper(S2, src, dst)
w_yellow = warper(yellow, src, dst)
w_white = warper(white, src, dst)
w_image = warper(image, src, dst)
w_binary =  warper(binary, src, dst)

## Plots
f, axarr = plt.subplots(3, 2, figsize=(20,18), sharex='col')
axarr[0,0].set_title('Gradient x', fontsize=20)
axarr[0,0].imshow(w_gradx, cmap='gray')
axarr[0,1].set_title('Gradient y', fontsize=20)
axarr[0,1].imshow(w_grady, cmap='gray')
axarr[1,0].set_title('Gradient magnitude', fontsize=20)
axarr[1,0].imshow(w_mag, cmap='gray')
axarr[1,1].set_title('Gradient direction', fontsize=20)
axarr[1,1].imshow(w_dir, cmap='gray')
axarr[2,0].set_title('Yellow', fontsize=20)
axarr[2,0].imshow(w_yellow, cmap='gray')
axarr[2,1].set_title('White', fontsize=20)
axarr[2,1].imshow(w_white, cmap='gray')
#axarr[3,0].set_title('S_gradx', fontsize=20)
#axarr[3,0].imshow(w_Sgradx, cmap='gray')
#axarr[3,1].set_title('S2', fontsize=20)
#axarr[3,1].imshow(w_S2, cmap='gray')
f.subplots_adjust(hspace=0.07, wspace=0)
#f.savefig('examples/thresholds_warped.jpg')


# Original image undistorted warped
# and final binary
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Original image warped', fontsize=20)
ax1.imshow(w_image)

ax2.set_title('Final output warped', fontsize=20)
ax2.imshow(w_binary, cmap='gray')
#f.savefig('examples/final_output.jpg')