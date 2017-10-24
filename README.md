# Advanced Lane Finding Project

## Resume
This project was awesome. An approach to lane lines detection is shown, where an image taken from a camera inside a car is processed to find lines and calculate the radius of curvature of the road and the offset distance of the car from the center of the lane.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The results turned out really successfully, as you can see in [this video](https://youtu.be/FSbxPJsh6zM)

[//]: # (Image References)

[undistorted]: ./camera_cal/test0.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[thresholds]: ./output_images/thresholds.jpg "Gradient and color thresholds"
[thresholds_warped]: ./output_images/thresholds_warped.jpg "Gradient and color thresholds warped"
[final_output]: ./output_images/final_output.jpg "Final binary output"
[image3]: ./output_images/binary_test5.jpg "Binary Example"
[image33]: ./output_images/combined_binary_test5.jpg "Combined binary Example"
[image4]: ./output_images/perspective_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/search_blind.jpg "Fit Visual"
[image55]: ./output_images/search_fast.jpg "Fit Visual"
[image6]: ./output_images/final_result.jpg "Output"
[video1]: ./output_project_video.mp4 "Video"
[R]: ./output_images/R.png "Radius of curvature"
[poly]: ./output_images/poly.png "Second order polynomial"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `camera_calibration.py`, that also uses a function `calibrate_camera()` from `utils.py` (lines 23-60)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted image][undistorted]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 62 through 273, in `utils.py`). 
As for the gradient, I used a combination of threshold ranges for the gradient in the 'x' direction, 'y' direction, the magnitude of the gradient and the direction of the gradient.
This exact combination is described in line 169 of the file `utils.py`:

    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

As for the color, I filter white and yellow channels from the image by transforming to HLS color space and then applying a cv2.inRange, with low and high thresholds taken from 
[this article (white case)](https://medium.com/@royhuang_87663/how-to-find-threshold-f05f6b697a00#.ebexlpkil)
and 
[this article (yellow case)](https://chatbotslife.com/advanced-lane-line-project-7635ddca1960#.uw016flos)

This values were the ones that worked best for me.
The functions that do this job are coded at lines 211 through 231.

In the following figure we can see the binary output for each gradient and color threshold type

![Gradient and color thresholds][thresholds]

In the following image we can see in the Green channel the contribution of the gradients and in the Blue channel the contribution of the S channel.

![Combined binary example][image33]

Here's an example of my output for this step.

![Binary example][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 302 through 312 in the file `utils.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. 
To calculate this points, I wrote a function taking the image size as an input in the following manner:

```
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

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 582, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1106, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped image][image4]

In the following figure, we can see all the binary outputs of the gradient and color thresholds warped with this values:

![Gradient and color thresholds warped][thresholds_warped]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify lane-line pixels I coded 2 functions: 

- a "blind search" algorithm, and 
- a "fast search" algorithm.

In the blind search algorithm I took a histogram of the bottom half of the binary warped output from the previous section. With this, I found the peaks of the left and right halves of the histogram, which were the starting points for the left and right lines.
Then, I draw a rectangle centered in those points and calculate the number of white pixels inside it. If there were more than 50 pixels in there, I updated the center of the rectangle to the mean point of all those pixels. Then I iterate, calculating the number of pixels inside the new window, going down through the y-axis. I divided the image in 9 rows, so the algorithm took 9 iterations for each input.
In every step, I stored the identified pixels for further processing. With this, it was possible to re-draw this pixels in color: left in red, right in blue.
After this, I fit my lane lines with a 2nd order polynomial for each line:

![Second order polynomial][poly]

and draw the resulting parabola in yellow.
In the following picture we can see the result of the blind search function.

![Blind search][image5]

For the following image in the pipeline, it is possible to take advantage of the previous information, considering that the coefficients of the polynomial won't change a lot. So for the "fast search" algorithm, I took the previous polynomial and found the pixels within a margin of +-100 pixels of that estimated lane lines, 
and fit again with a 2nd order polynomial using these new pixels, 
as we can see in the output of that function:

![Fast search][image55]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature at some point *y* for a second order polynomial can be computed with the following formula:

![R][R]

For this application, I evaluated this formula at the bottom of the image, so 

    y = image.shape[0] 

Furthermore, I took a moving average of the last 30 calculated radius of curvature to smooth a little bit this measurement.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 567 through 604 in my code in `utils.py` in the functions `visualize_unwraped()` and `put_text()`.  Here is an example of my result on a test image:

![Final output][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/zItOA7QE9b4)


### Challenge

As a challenge, I carried on almost the same algorithm for a video of a different road. In this case, I always performed the "blind search" algorithm to find lane lines. This way I achieved better results.

Here's a [link to my video result](./output_challenge_allblind_video.mp4)



---

### Discussion and future work

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The result in the challenging video is worst, because all the thresholds and parameters where adjusted analyzing images of the first project video. To achieve better results, it would be necessary to re-adjust these parameters with images from the challenging video, or trying different combinations of gradient and color.
There are also other methods to make the algorithm more robust, and they should be researched for future work on this project.