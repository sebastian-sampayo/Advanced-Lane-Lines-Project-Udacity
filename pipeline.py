# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 4: Advance Lane Lines
# Date: 26th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: pipeline.py
# =========================================================================== #
# Pipeline for image processing

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from utils import *
#from Camera import *
from Line import *

# --------------------------------------------------------------------------- #
camera_calibration_filename = 'camera_cal/mtx_dist_pickle.p'
input_filename = 'test_images/test5.jpg'
output_filename = 'output_images/binary_test6.jpg'
perspective_filename = 'output_images/perspective_straight_lines2.jpg'

# --------------------------------------------------------------------------- #
# Load camera calibration matrix and distortion coefficients
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']
# cam = Camera()
# cam.load(camera_calibration_filename)
# mtx = cam.mtx
# dist = cam.dist

# Read in a tests image
image = mpimg.imread(input_filename)

# Undistort test image
undistorted = cv2.undistort(image, mtx, dist, None, mtx)

# Binary output
sobel_kernel = 3
thresh = [
         (20, 100), 
         (30, 100), 
         (0.7, 1.3), 
         (170, 255)
         ]
binary = color_and_gradient(undistorted, sobel_kernel=sobel_kernel, 
                            thresh=thresh, verbose=False)

# --------------------------------------------------------------------------- #
# Plotting thresholded images
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('Original image', fontsize=30)
ax1.imshow(image)

ax2.set_title('Undistorted image', fontsize=30)
ax2.imshow(undistorted)

ax3.set_title('Binary image', fontsize=30)
ax3.imshow(binary, cmap='gray')
# f.savefig(output_filename)


# --------------------------------------------------------------------------- #
## Perspective transform
# src: 4 points in the photo that should be a rectangle in the real world.
# dst: The 4 vertices of a rectangle in the transformed (output) image.
img_size = (image.shape[1], image.shape[0])
src, dst = points_for_warper(img_size)
# Lines for source polygon
lines = lines_from_points(src)

# Plot lines on binary unwarped image
binary_lines = np.dstack(( binary, binary, binary ))
draw_lines(binary_lines, lines)

# Plot lines on undistorted image
undist_lines = normalize(undistorted)
draw_lines(undist_lines, lines)

# --------------------------------------------------------------------------- #
# Warp binary and undistorted images
binary_warped = warper(binary, src, dst)
undistorted_warped = warper(undistorted, src, dst)

# --------------------------------------------------------------------------- #
# Lines for destination rectangle
lines = lines_from_points(dst)

# Plot lines on binary warped image
binary_warped_lines = np.dstack(( binary_warped, binary_warped, binary_warped ))
draw_lines(binary_warped_lines, lines)

# Plot lines on undistorted warped image
undistorted_warped_lines = normalize(undistorted_warped)
draw_lines(undistorted_warped_lines, lines)

# --------------------------------------------------------------------------- #
## Plots
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Source region for transformation', fontsize=30)
ax1.imshow(undist_lines)

ax2.set_title('Warped image', fontsize=30)
ax2.imshow(undistorted_warped_lines)
#f.savefig(perspective_filename)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Source region for transformation', fontsize=30)
ax1.imshow(binary_lines)

ax2.set_title('Warped image', fontsize=30)
ax2.imshow(binary_warped_lines)
#f.savefig(perspective_filename)


# --------------------------------------------------------------------------- #
# Find and plot lane lines
left_line = Line()
right_line = Line()
out_img = find_lines_blind(binary_warped, left_line, right_line)
visualize_lines_search_blind(binary_warped, left_line, right_line, out_img)
R = radius_of_curvature(left_line, right_line)
offset = center_offset(img_size, left_line, right_line)
print('Radius: {}m'.format(int(R)))
print('Offset: {}m'.format(offset))
find_lines_fast(binary_warped, left_line, right_line)
visualize_lines_search_fast(binary_warped, left_line, right_line)

M = cv2.getPerspectiveTransform(src, dst)
Minv = np.linalg.inv(M)
left_fit = left_line.current_fit
right_fit = right_line.current_fit

result = visualize_unwraped(undistorted, binary_warped, left_fit, right_fit, Minv)

R = radius_of_curvature(left_line, right_line)
offset = center_offset(img_size, left_line, right_line)
result = put_text(result, R, offset)

plt.figure()
plt.imshow(result)
#plt.savefig('output_images/final_result.jpg')