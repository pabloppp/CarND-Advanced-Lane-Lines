## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_comparison.jpg "calibration"
[image2]: ./output_images/undistorted_comparison.jpg "undistorted"
[image3]: ./output_images/binary_threshold_comparison.jpg "threshold"
[image4]: ./output_images/birdseye_comparison.jpg "birdseye"
[image5]: ./output_images/pipeline_comparison.jpg "threshold+ birdseye"
[image6]: ./output_images/lanes_comparison.jpg "lane detection"
[image7]: ./output_images/highlighted_comparison.jpg "lane highlight"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibrarion is contained in [utils/calibration.py](src/utils/calibration.py) and [calibrate.py](src/calibrate.py)

The `get_distortion_matrix` function accepts as a parameter a path for the calibation images `../camera_cal/calibration*.jpg`. We then initialize the object and extract the and image points from the ChessboardCorners in the calibration images.
Finally, the calibrateCamera will give us both the distortion and calibration matrix.

We only need to call `cv2.undistort` passing both of this things to undistort an image using the obtained values.
I decided to save those to [calibration_matrix.p](src/calibration_matrix.p) to be able to use it easily in future steps.

I also created the `setup_undistort` function that loads in memory those parameters and returns a lambda that we can call to aply the undistort easily.

Here's a comparison of the undistortion applied to te chessboard:
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
I created the [test calibration matrix.py](src/test_calibration_matrix.py) to be able to see how this correction affects the test images.

If we apply the distortion correction to the road images, te effect is much less noticeable, but we know that any measurement performed will be much more accurate:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I created the [test thresholded.py](src/test_thresholded.py) to be able to see how this threshold affects the test images. 
Check also [utils/highlight lane.py](src/utils/highlight_lane.py) to see the thresold values I used.

I did a lof of testing around this step, because it's the more important of the whole pipeline. If we manage to filter out everything except for the lane lines, the rest of the pipeline will work seamlessly.

After doing some research I decided to attempt combinig ALL the learn filters, I used the HSL color space to filter the **white** and **yellow** from te images, this will give us a good enough threshold, working 90% of the time. But in order to detect the lines the remaining 10% of the time, I did a combination of X/Y Sobel threshold, Absolute Sobel Threshold and Angular Sobel threshold (the X Threshold gives the best results, the rest are just there to slightly improve the results).

After tweaking the values for a long time attempting to get the best results in all the test images, this is the result I obtained:

![alt text][image3]

You can notice there's still some noise on the lanes, this was necessary in order for the thresold to also work in mutiple conditions (shadow, white lane, etc...). This will be corrected in the lane detection step.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I created the [test birds eye view.py](src/test_birds_eye_view.py) to be able to see how this perspective transform affects the test images. 
Check also the function `birds_eye_view` in [utils/image pipeline.py](src/utils/image_pipeline.py) to see the transformation shapes I used.

This code is very simple, I just carefuly found the position of 4 pixels in the image that formed a parallelogram contaning the lane and translated them to a fixed-shaped rectangle using the function `cv2.getPerspectiveTransform` to generate the tranformation matrix (I also generated the inverse matrix in order to unwarp the image for lane projection later.)

The pixel positions I used are:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 215, 706      | 320, 0        | 
| 580, 460      | 320, 720      |
| 704, 460     | 960, 720      |
| 1094, 706      | 960, 0        |

Here's the transformation result:
![alt text][image4]

And here is the same transformation applied to a thresholded image (I also drew the transformation shapes for debug purposes)

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I created the [test lane detection.py](src/test_lane_detection.py) to be able to see how the lane detectionworks on test images. 
Check also [utils/lane detection.py](src/utils/lane_detection.py) to see the full lane detection code.

This part, even though it's not the most crucial for lane detection, was the hardes to understand. I wen through the example code a few times and did some research online until I felt confortable enough to implement it myself.

I didn't change much how the example code was working, I extracted the relevant utility functions (like `_find_window_centroids`) and used them inside the `detect_lanes` functions.
I decided to use a window of 70x90 with a margin of 120, in order to detect with accuracy the right white lane (not continuous). Because of the big height of the window, we will find fewer points to generate the polynomials, but in exchange we will filter out a lot of the noise from the thresholded images.

I also added a vertical threshold of 2000 / window_dims[1] to avoid te big blobs (like when the lane is white)

My first approach was very simple, it wasn't until the end that I added three things to improve the detected lanes (and they work very well):
- First of all, as sugested in the excercice, I saved the last polynomial detected to speed up the following calculations (because we already know were the lane is, we can use this to look for the lane in the next frame).
- Second I did some error discrimination: if the radius detected from the 2 lines was very different (an error of 180% or more) I discard this reading and use the latests correct lane reading for following detections.
- Finally, I decided to smooth the detected lane by doing an average of the correctly detected lane in the last 4 frames (asuming that the video has 30 frames per second, this adds a small lag of 130ms so if the card did a very unexpected turn, it will take this time for it to adapt to the road, but for this exercice it's ok)

Here's an example of the detected lanes in a test image:
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Check the methods `_curve_radius` and `_offcenter_position` in [utils/lane detection.py](src/utils/lane_detection.py) to see the full implementation.

I didn't change much from the proposed implementation. I used the proposed correction values `ym_per_pix = 30 / 720` and `xm_per_pix = 3.7 / 640` and then just applied the equations.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Check the method `_lane_shape ` in [utils/lane detection.py](src/utils/lane_detection.py) to see the full implementation of how the lane polygon is generated.

See also [test highlight lane.py](src/test_highlight_lane.py) and  and [utils/lane detection.py](src/utils/highlight_lane.py) to a complete pipeline of how the lane highlight is drawn on top of the image, the steps are the following:
- Undistorting the image using `undistort(img)`
- Extract threshold and bird-eye view using `pipeline(img)`
- Detect the lanes and generate the highlight polygon using `detect_lanes(img)`
- Unwarp the image from bird-eye to persoecrtive using `warp(img)` using the inverse warp matrix
- combine the original undistorted image + the highlight polygon with `cv2.addWeighted` to add some transparency and be able to see te road behind.

This is the result:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Check the full video pipeline in [generate video.py](src/generate_video.py). In order to keep state between frames and organizing the video pipeline methods I created the class [LaneProcessor](src/utils/lane_processor.py). I also created the class [Lane](src/utils/lane.py) that keeps some of the state required for the improved lane detection and adds the helper methods for smoothing the results.

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I mentioned before, I believe the key in this whole process is the binary threshold. 

I achieved good enough results for the exercice video, but they won't work for the challenge videos. There might be a way to fine-tune the tresholding so it works better in different light/weather conditions, but this has it's limitations on how muche we're able to generalize and will eventually fail.

Also, this whole image processing is very slow, if we needed to detect the lanes in real-time we would need a faster way.

I believe that a possible approach for this would be using deep learning for feature extraction, if we had labeled images with the lane highlisted in various light/weather conditions we could train a model to predict where the lane is in a much optimal way that what we can achieve using only computer vision.

#### A note about the smoothing:   
I averaged the last 4 frames to smooth the lanes, this works well for this exercice but won't work in real life because of the lag.   
For this exercice though I noticed an interesting effect of this: when the can goes through some bumps, the image's vertical shift changes. When performing the birds-eye transform we asume the vertical shift is the same, though the lane lines should be parallel but when it changes the lanes won't be parallel anymore so the readings will be off. 
The smoothing accidentally corrects for this, giving an actually more acurate depiction of the road if the shift hadn't changed, but because it has for a few milliseconds it seems like the highlighted lane is not quite right, when it actually is!

Thanks for reading, it took me a lot of effort to finish this in such a short ammount of time so a lot of assumptions might be wrong and a lot of things could be improved, if you find this by accident and have some suggestion pleas create an issue in the repo! 
