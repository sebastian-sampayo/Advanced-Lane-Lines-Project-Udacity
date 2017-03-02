# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 4: Advance Lane Lines
# Date: 26th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: utils.py
# =========================================================================== #
# Utility functions

import numpy as np
import cv2
import matplotlib.pyplot as plt

from Line import *

# --------------------------------------------------------------------------- #
# Takes a list of images and calculates the calibration matrix and distortion
# coefficients. These images are photos of a known chessboard pattern of size:
# nx x ny. img_size is the size of the images to undistort in the application.
# Verbose mode shows the detected points in the input images.
def calibrate_camera(images, nx, ny, img_size, verbose=False):
  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  objp = np.zeros((nx*ny,3), np.float32)
  objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d points in real world space
  imgpoints = [] # 2d points in image plane.

  # Make a list of calibration images
  # images = glob.glob('../camera_cal/calibration*.jpg')

  # Step through the list and search for chessboard corners
  for fname in images:
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      # Find the chessboard corners
      ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

      # If found, add object points, image points
      if ret == True:
          objpoints.append(objp)
          imgpoints.append(corners)

          if verbose:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

  if verbose:
    cv2.destroyAllWindows()
    
  # Do camera calibration given object points and image points
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

  return mtx, dist
  
# --------------------------------------------------------------------------- #
# Calculate directional gradient, in 'x' or 'y' orientation, using Sobel operator.
# Output is a binary image where a pixel is '1' only if the gradient at that 
# point is inside the threshold range.
# Threshold must be between (0, 255)
# Input image must be in RGB format
def abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(20, 100)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

# --------------------------------------------------------------------------- #
# Calculate gradient magnitude.
# $$ abs_{sobel} = \sqrt{sobel_x^2 + sobel_y^2} $$
# Output is a binary image where a pixel is '1' only if the gradient magnitude 
# at that point is inside the threshold range.
# Threshold must be between (0, 255)
# Input image must be in RGB format
def mag_threshold(image, sobel_kernel=3, thresh=(30, 100)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output

# --------------------------------------------------------------------------- #
# Calculate gradient direction
# Output is a binary image where a pixel is '1' only if the gradient direction 
# at that point is inside the threshold range.
# Threshold must be between (0, np.pi/2)
# Input image must be in RGB format
def dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    scaled_sobel = grad_dir
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_dir)
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output

# --------------------------------------------------------------------------- #
# Combined gradients
# Output is a binary image where a pixel is '1' only if the combined gradients
# at that point is inside the threshold range.
# Threshold order:
# 0: directional gradient
# 1: gradient magnitude
# 2: gradient direction
# Threshold example: thresh = [(20, 100), (30, 100), (0.7, 1.3)]
# Choose a larger odd number to smooth gradient measurements
# Input image must be in RGB format
def combined_gradients(image, sobel_kernel=3, thresh=[(20, 100), (30, 100), (0.7, 1.3)]):
  # Separate thresholds
  thresh_grad = thresh[0]
  thresh_mag = thresh[1]
  thresh_dir = thresh[2]
  
  # Apply each of the thresholding functions
  gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=sobel_kernel, thresh=thresh_grad)
  grady = abs_sobel_threshold(image, orient='y', sobel_kernel=sobel_kernel, thresh=thresh_grad)
  mag_binary = mag_threshold(image, sobel_kernel=sobel_kernel, thresh=thresh_mag)
  dir_binary = dir_threshold(image, sobel_kernel=sobel_kernel*5, thresh=thresh_dir)
  
  # Combination
  combined = np.zeros_like(dir_binary)
  combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
  
  return combined

# --------------------------------------------------------------------------- #
# HLS Saturation threshold
# Input image must be in RGB format
def S_threshold(image, thresh=(170, 255)):
  # Convert to HLS color space and separate the S channel
  # Note: img is the undistorted image
  hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]
  
  # Threshold color channel
  s_thresh_min = thresh[0]
  s_thresh_max = thresh[1]
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
  return s_binary
  
# --------------------------------------------------------------------------- #
# RGB Red threshold
# Input image must be in RGB format
def R_threshold(image, thresh=(250, 255)):
  R_channel = image[:,:,0]
  
  # Threshold color channel
  thresh_min = thresh[0]
  thresh_max = thresh[1]
  R_binary = np.zeros_like(R_channel)
  R_binary[(R_channel >= thresh_min) & (R_channel <= thresh_max)] = 1
    
  return R_binary
    
# --------------------------------------------------------------------------- #
# Normalize image
def normalize(image):
  maxval = np.max(image)
  output = image/maxval
  return output
  
# --------------------------------------------------------------------------- #
# Yellow threshold
# Input image must be in RGB format
# Thresholds work best from:
# https://chatbotslife.com/advanced-lane-line-project-7635ddca1960#.uw016flos
def yellow_threshold(image, thresh_min=(0, 100, 100), thresh_max=(50, 255, 255)):
  HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  yellow = cv2.inRange(HLS, thresh_min, thresh_max)
  yellow = normalize(yellow)
  return yellow

# --------------------------------------------------------------------------- #
# White threshold
# Input image must be in RGB format
# Thresholds work best from:
# https://medium.com/@royhuang_87663/how-to-find-threshold-f05f6b697a00#.ebexlpkil
def white_threshold(image, thresh_min=(10, 200, 150), thresh_max=(40, 255, 255)):
  HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  white = cv2.inRange(HLS, thresh_min, thresh_max)
  white = normalize(white)
  return white

# --------------------------------------------------------------------------- #
# Combine color and gradient thresholds to output a binary image for lane line detection
# Input image must be in RGB format
# Threshold order:
# 0: directional gradient
# 1: gradient magnitude
# 2: gradient direction
# 3: HLS Saturation
def color_and_gradient(image, sobel_kernel=3, thresh=[(20, 100), (30, 100), (0.7, 1.3), (170, 255)], verbose=False):
  thresh_grad = thresh[:3]
#  thresh_color = thresh[3]
  grad = combined_gradients(image, sobel_kernel, thresh_grad)
#  color = S_threshold(image, thresh_color)
  
  # Yellow filter
  yellow = yellow_threshold(image)
  
  # White filter
  white = white_threshold(image)
  
  color = np.zeros_like(grad)
  color[(yellow == 1) | (white == 1)] = 1
    
  # Combine the two binary thresholds
  combined_binary = np.zeros_like(grad)
  combined_binary[(color == 1) | (grad == 1)] = 1

  if verbose:
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(grad), grad, color))
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Stacked thresholds', fontsize=20)
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient thresholds', fontsize=20)
    ax2.imshow(combined_binary, cmap='gray')
#    f.savefig('examples/combined_binary_test5.jpg')
  
  return combined_binary

# --------------------------------------------------------------------------- #
# Source and destination points for perspective transform
def points_for_warper(img_size):
  src = np.float32(
      [[(img_size[0] / 2) - 58, img_size[1] / 2 + 100],
      [((img_size[0] / 6) - 10), img_size[1]],
      [(img_size[0] * 5 / 6) + 40, img_size[1]],
      [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
      
  dst = np.float32(
      [[(img_size[0] / 4), 0],
      [(img_size[0] / 4), img_size[1]],
      [(img_size[0] * 3 / 4), img_size[1]],
      [(img_size[0] * 3 / 4), 0]])
      
  return src, dst

# --------------------------------------------------------------------------- #
# Create a list with start and end points for lines to draw, from vertices.
def lines_from_points(p):
  lines = [[ p[0], p[1] ],
         [ p[1], p[2] ],
         [ p[2], p[3] ],
         [ p[3], p[0] ],
         ]
  return lines
  
# --------------------------------------------------------------------------- #
# Compute and apply perspective transform.
# src: 4 points in the photo that should be a rectangle in the real world.
# dst: The 4 vertices of a rectangle in the transformed (output) image.
# Returns the warped image
def warper(img, src, dst):
  img_size = (img.shape[1], img.shape[0])
  M = cv2.getPerspectiveTransform(src, dst)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

  return warped
  
# --------------------------------------------------------------------------- #
def draw_lines(img, lines, color=[1, 0, 0], thickness=2):
  """
  This function draws `lines` with `color` and `thickness`.    
  Lines are drawn on the image inplace (mutates the image).
  If you want to make the lines semi-transparent, think about combining
  this function with the weighted_img() function below
  """
  for line in lines:
    p1 = line[0]
    p2 = line[1]
    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness)

# --------------------------------------------------------------------------- #    
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
# --------------------------------------------------------------------------- #    
# This function implements the "blind search" algorithm to find lane lines
def find_lines_blind(binary_warped, left_line, right_line):
  # Take a histogram of the bottom half of the image
  histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
  # Create an output image to draw on and  visualize the result
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
  
  # Find the peak of the left and right halves of the histogram
  # These will be the starting point for the left and right lines
  midpoint = np.int(histogram.shape[0]/2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint

  # Choose the number of sliding windows
  nwindows = 9
  # Set height of windows
  window_height = np.int(binary_warped.shape[0]/nwindows)
  # Identify the x and y positions of all nonzero pixels in the image
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  # Current positions to be updated for each window
  leftx_current = leftx_base
  rightx_current = rightx_base
  # Set the width of the windows +/- margin
  margin = 100
  # Set minimum number of pixels found to recenter window
  minpix = 50
  # Create empty lists to receive left and right lane pixel indices
  left_lane_inds = []
  right_lane_inds = []

  # Step through the windows one by one
  for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = binary_warped.shape[0] - (window+1)*window_height
      win_y_high = binary_warped.shape[0] - window*window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      # Draw the windows on the visualization image
      cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
      cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
          leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
          rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  # Concatenate the arrays of indices
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds] 

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)
  
  # Calculate the new radii of curvature
  
  # Update line objects
  left_line.current_fit = left_fit
  right_line.current_fit = right_fit
  
  left_line.allx = leftx
  left_line.ally = lefty
  
  right_line.allx = rightx
  right_line.ally = righty
  
  y_eval = binary_warped.shape[0]
  left_line.update_radius(y_eval)
  right_line.update_radius(y_eval)
  
  return out_img
  
# --------------------------------------------------------------------------- #
# Calculate radius of curvature
# Actually, just take the average of left and right radius of curvature
def radius_of_curvature(left_line, right_line):
  left_curverad = left_line.radius_of_curvature
  right_curverad = right_line.radius_of_curvature

  # Average left and right radius of curvature
  R = (left_curverad + right_curverad) / 2
  
  return R
  
# --------------------------------------------------------------------------- #
# Calculate center offset, i.e. the distance of the center of the car
# from the center of the lane.
def center_offset(img_size, left_line, right_line, xm_per_pix = 3.7/700):
  left_fit = left_line.current_fit
  right_fit = right_line.current_fit
  
  # Calculate the x value of the left and right lines
  y_eval = img_size[1]
  x_left = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
  x_right = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

  # Assuming the camera is centered at the middle width of the car:
  offset = ((x_left + x_right)/2 - img_size[0]/2 )* xm_per_pix

  return offset
  
# --------------------------------------------------------------------------- #
# Visualize line search
def visualize_lines_search_blind(binary_warped, left_line, right_line, out_img):
  left_fit = left_line.current_fit
  right_fit = right_line.current_fit
  leftx = left_line.allx
  lefty = left_line.ally
  rightx = right_line.allx
  righty = right_line.ally

  # Generate x and y values for plotting
  ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  out_img[lefty, leftx] = [255, 0, 0]
  out_img[righty, rightx] = [0, 0, 255]

  plt.figure()
  plt.imshow(normalize(out_img))
  plt.plot(left_fitx, ploty, color='yellow')
  plt.plot(right_fitx, ploty, color='yellow')
  plt.xlim(0, 1280)
  plt.ylim(720, 0)
  
#  plt.savefig('examples/search_blind.jpg')

# --------------------------------------------------------------------------- #  
# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
def find_lines_fast(binary_warped, left_line, right_line):
  left_fit = left_line.current_fit
  right_fit = right_line.current_fit

  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  margin = 100
  left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
  right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
  
  # Again, extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)
  
  # Update line objects
  left_line.current_fit = left_fit
  right_line.current_fit = right_fit
  
  left_line.allx = leftx
  left_line.ally = lefty
  
  right_line.allx = rightx
  right_line.ally = righty
  
  y_eval = binary_warped.shape[0]
  left_line.update_radius(y_eval)
  right_line.update_radius(y_eval)
  
  
# --------------------------------------------------------------------------- #
# Visualize line search
def visualize_lines_search_fast(binary_warped, left_line, right_line):
  left_fit = left_line.current_fit
  right_fit = right_line.current_fit
  leftx = left_line.allx
  lefty = left_line.ally
  rightx = right_line.allx
  righty = right_line.ally

  margin = 100

  # Generate x and y values for plotting
  ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  # Create an image to draw on and an image to show the selection window
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
  window_img = np.zeros_like(out_img)
  # Color in left and right line pixels
  out_img[lefty, leftx] = [255, 0, 0]
  out_img[righty, rightx] = [0, 0, 255]
  
  # Generate a polygon to illustrate the search window area
  # And recast the x and y points into usable format for cv2.fillPoly()
  left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
  left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
  left_line_pts = np.hstack((left_line_window1, left_line_window2))
  right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
  right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
  right_line_pts = np.hstack((right_line_window1, right_line_window2))
  
  # Draw the lane onto the warped blank image
  cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
  cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
  result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
  plt.figure()
  plt.imshow(normalize(result))
  plt.plot(left_fitx, ploty, color='yellow')
  plt.plot(right_fitx, ploty, color='yellow')
  plt.xlim(0, 1280)
  plt.ylim(720, 0)
#  plt.savefig('examples/search_fast.jpg')

# --------------------------------------------------------------------------- #
# Unwrap the lane mask to the original undistorted image
def visualize_unwraped(image, binary_warped, left_fit, right_fit, Minv):
  ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  # Create an image to draw the lines on
  warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
  
  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))
  
  # Draw the lane onto the warped blank image
  aux = np.int_([pts])
  cv2.fillPoly(color_warp, aux, (0,255, 0))
  
  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
  # Combine the result with the original image
  result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
  return result

# --------------------------------------------------------------------------- #  
# Add legends to the image (radius of curvature and center offset)
def put_text(img, R, offset):
  text = 'Radius of Curvature: {}m'.format(int(R))
  text2 = 'Center offset {:.2f}m'.format(offset)
  text_pos = (50, 100)
  text2_pos = (50, 200)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 2
  cv2.putText(img, text, text_pos, font, font_scale, (255,255,255), 2, cv2.LINE_AA)
  cv2.putText(img, text2, text2_pos, font, font_scale, (255,255,255), 2, cv2.LINE_AA)
  return img
  
# --------------------------------------------------------------------------- #
# Final Pipeline
# Input:
# mtx, dist
# left_line, right_line
# blind: A flag to determine whether to use the "blind" or the "fast" search algorithm
def pipeline(image, mtx, dist, left_line, right_line, blind=False):
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
  
  # Wrap binary image
  img_size = (image.shape[1], image.shape[0])
  src, dst = points_for_warper(img_size)
  binary_warped = warper(binary, src, dst)

  # Determine which search algorithm to use
  if blind:
    blind_img = find_lines_blind(binary_warped, left_line, right_line)
  else:
    find_lines_fast(binary_warped, left_line, right_line)

  left_fit = left_line.current_fit
  right_fit = right_line.current_fit

  # Draw the detected lane region on the original undistorted image.
  M = cv2.getPerspectiveTransform(src, dst)
  Minv = np.linalg.inv(M)
  result = visualize_unwraped(undistorted, binary_warped, left_fit, right_fit, Minv)
  return result