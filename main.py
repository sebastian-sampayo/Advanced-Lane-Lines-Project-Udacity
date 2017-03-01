# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 4: Advance Lane Lines
# Date: 26th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: main.py
# =========================================================================== #
# Main file
'''
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

# --------------------------------------------------------------------------- #
from moviepy.editor import VideoFileClip
import pickle
import numpy as np

from utils import *
from Line import *

# --------------------------------------------------------------------------- #
input_video_filename = 'project_video.mp4'
output_video_filename = 'output_' + input_video_filename
camera_calibration_filename = 'camera_cal/mtx_dist_pickle.p'

# --------------------------------------------------------------------------- #
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']

# Initialize lane lines
left_line = Line()
right_line = Line()

# This counter is for detecting the first processed image
counter = 0

# A list to accumulate radius of curvature values and the take the average.
N_avg = 30
R_list = []
for i in range(N_avg):
  # Init with 1km.
  R_list.append(1000)

# This is the function that will process each image in the video
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    
    # Use the blind search in the first image only.
    global counter
    if counter == 0:
      blind = True
      print('BLIND!')
    else:
      blind = False
#    blind = True
      
    # Pipeline!
    result = pipeline(image, mtx, dist, left_line, right_line, blind)
    
    # Accumulate the radius of curvature and take the mean value
    R = radius_of_curvature(left_line, right_line)
    R_list.pop()
    R_list.insert(0, R)
    R_avg = np.mean(R_list)
    
    # Calculate the center offset
    img_size = (image.shape[1], image.shape[0])
    offset = center_offset(img_size, left_line, right_line)
    
    # Put the legend in the final output (radius and offset)
    result = put_text(result, R_avg, offset)

    counter += 1
    
    return result
    
# Load video and process every image.
clip1 = VideoFileClip(input_video_filename)
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output_video_filename, audio=False)