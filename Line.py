# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 4: Advance Lane Lines
# Date: 26th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: Line.py
# =========================================================================== #
# Line class

import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  # left_fit
        #radius of curvature of the line in meters
        self.radius_of_curvature = None # left_R
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  # leftx
        #y values for detected line pixels
        self.ally = None  # lefty
        #meters per pixel in x dimension
        self.xm_per_pix = 3.7/700
        #meters per pixel in y dimension
        self.ym_per_pix = 30/720
        
    def update_radius(self, y_eval):
      allx = self.allx
      ally = self.ally
      
      # Fit a second order polynomial to each
      fit = np.polyfit(ally*self.ym_per_pix, allx*self.xm_per_pix, 2)    
      
      # Calculate the new radii of curvature
      radius = ((1 + (2*fit[0]*y_eval*self.ym_per_pix + fit[1])**2)**1.5) / np.absolute(2*fit[0])
      self.radius_of_curvature = radius
      
      
    def update_pos(self, img_size):
      fit = self.current_fit
      
      y_eval = img_size[1]
      x_pos = fit[0]*y_eval**2 + fit[1]*y_eval + fit[2]
    
      offset = (x_pos - img_size[1]/2 ) * self.xm_per_pix
      self.line_base_pos = offset
