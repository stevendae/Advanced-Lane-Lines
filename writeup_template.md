## Advanced Lane Lines README

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/test3/Original_Image.jpg "Original"
[image3]: ./output_images/test3/Distortion_Corrected_Image.jpg "Undistorted"
[image4]: ./output_images/test3/Thresholded_Binary_Image.jpg "Thresholded Binary Image"
[image5]: ./output_images/test3/Perspective_Transform.jpg "Birds Eye View Image"
[image6]: ./output_images/test3/Lane_Pixels_and_Fitted_Line.jpg "Lane Pixels and Polyfit"
[image7]: ./output_images/test3/Final_Result.jpg "Final Result with Lane Marking, Curvature, and Offset"
[image8]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook titled "Pipeline_Test_Images" located in "Pipeline_Test_Images.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the global space. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### The code that was developed to realize the objectives in this section (single images) is titled "Pipeline_Test_Images.ipynb"

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one. The first image is the test image titled test3.jpg in the test_images folder. The second is it's distortion corrected equivalent.:

##### Original Image
![alt text][image2]

##### Undistorted Image
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. This function can be found in the code cell titled "Color and Gradient Threshold". 

| Threshold       | Purpose   | 
|:-------------:|:-------------:| 
| Directional Gradient    | Apply Sobel Operator in Either X or Y Direction     | 
| Gradient Magnitude    | Combine Sobel Operator Product of X and Y      |
| Gradient Direction  | Identify Edges Oriented At Particular Angle     |
| Saturation Channel    | Identify Single Colour Objects Regardless of Lightness/Shadow    |
| Grayscale Channel    | Identify Lane Lines via Bright Salient Appearance   |

The final binary result was achieved by incorporating the pixels that are within the threshold ranges for both x and y gradients or both gradient magnitude and direction. Both grayscale and saturation channels were considered in the final output to maximize visibility.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in the 9th code cell of the IPython notebook.  The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. The function excercises two Open.CV2 methods which include cv2.getPerspectiveTransform and cv2.warpPerspective. The former method returns a transform matrix that allows the image to warp into a birds eye perspective and the latter method performs the warp computation. I chose the source and destination points by method of trial and error, and from insight from my working community until I achieved a result where the lane lines appeared relatively parallel.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 190, 720        | 
| 548, 480      | 190, 0      |
| 740, 480     | 1130, 0      |
| 1130, 720      | 1130, 720        |

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The process in which I was able to identify the lane-line pixels and fit their positions with a polynomial is described in the functions titled 'find_lines' and 'linefit'. They are located in the 10th and 11th code cells, respectively. In the 'find_lines' function I begin by creating a histogram to detect the number of pixels there are in every column on the x-axis. The two peaks with the most points are the indicators that signify the base of the lanes. I then excercised a sliding window technique in which begins by creating a small rectangular window with the bottom touching the x-axis and midpoint of the base of the window positioned on the basepoint of the lane; for both lanes. I then store the indices of the pixels that exist inside the window and concatenate the list of indices to my 'global' list which indicates 'all' the indices occupied by each lane pixel. If the number of pixels counted in each window exceeds 50 then a new base is calculated which takes the mean of all the x coordinates of the pixels in the window. This process is repeated, as the windows are stacked on top of each other, until the windows reach the top of the frame. This is calculated for both the left and right lane. The numpy polyfit function is then used to calculate the polynomial coefficients of the second order polynomial for the left and right lanes. 

The polynomial coefficients are then used as the input to the 'linefit' function where it returns the x coordinate for every y coordinate of the fitted line. This array is created to plot the points of the fitted line. 

The result of both functions can be seen in the image below. Where the windows are in green, the left lane pixels are in red, the right lane pixels are in blue, and the fitted lines indicating the shape of the lane is in yellow.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane and the offset position of the vehicle are calculated in the code cell titled 'Curvature' and 'Offset' in code cells number 12 and 13. The curvature is calculated by implementing the formulation described on this [page] (http://www.intmath.com/applications-differentiation/8-radius-curvature.php). The code to calculate the radius is described in the following section. The conversion factors are provided in the project description where they are originally extracted from the physical dimensions of the width of a lane and the length of a lane line, 30m and 3.7m respectively.

```python
def Curvature(ploty, leftx, rightx):    
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #radius = (left_curverad+right_curverad)/2 
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    return left_curverad, right_curverad
```
The result provides the radius of a curve in metres.

The offset of the car's position relative to the center of the lane is calculated in the code cell titled 'Offset'. This calculation begins in similar fashion with the 'find_lines' function where a histogram is used to calculate the x coordinates of the base of the lane lines. Once the bases are identified they are averaged to identify the midpoint of the lane. The midpoint is subtracted from the midpoint of the frame (assumed to be center of car), to get the offset distance in pixels. This value is then multiplied by the pixel to meter conversion factor of 0.0041. The conversion factor was a result of calculating an average ratio from the width of the lane (3.7m) to its pixel-equivalent length from the test images set. 

```python
def Offset(top_down):
    
    histogram = np.sum(top_down[top_down.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((top_down, top_down, top_down))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    midlane = (leftx_base+rightx_base)/2
    midcar = (top_down.shape[1])/2
    px_2_m = 0.004111
    
    offset = px_2_m*(np.absolute(midlane-midcar))
    
    return offset
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
