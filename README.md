##CarND Term1 P5 Vehicle Detection and Tracking

###A software is designed to process road video stream and track the positions of cars nearby.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[SDD_image0]: ./doc/tracking_design_diagram.png
[image1]: ./doc/car_not_car.png
[image2]: ./doc/HOG_example.jpg
[image3]: ./doc/car_detection_windows_multi_sizes.png
[image4_pipeline_eg1]: ./doc/heated_test1.jpg
[image4_pipeline_eg2]: ./doc/heated_test3.jpg
[image4_pipeline_eg3]: ./doc/heated_test4.jpg
[image5]: ./doc/bboxes_and_heat.png
[image6]: ./doc/labels_map.png
[image7]: ./doc/output_bboxes.png
[video1]: ./project_video.mp4


---
###Design

The following diagram shows the design of software archietecture and logic flow.

![alt text][SDD_image0]

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function get_hog_features() and getFeatures() at `FeatureExtractor_Hog.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. HOG parameters.

I tried various combinations of parameters and feed them to train the model. The following parameters shows a satisfactory performance in my testing: 

| Prameter        | Value         |
| --------------- |:-------------:|
| Color space     | YCrCb         |
| orientations    | 9             |
| pixels_per_cell | 8             |
| cell_per_block  | 2             |
| spatial_shape   | (32, 32)      |
| histogram bins  | 32            |

In general, pixels_per_cell captures the key car features, such as lamps and windows, in proportion to the  64x64 training data. The color space isolates out the illumination effection, so lighting variaion can be supressed in the model. The other parameters more or less provide a lot more data or details of the feature to be fed for model training. The trade-off is that smaller feature vector speeds up the video processing, while a bigger vector improves model prediction accuracy.

####3. Classifier

I trained a linear SVM using HOG and color features. In order to speed up the process, HOG vector is extracted over the whole video frame and then mapped to each small window. The color feature is extracted from direct pixel values of a size-reduced sub-image, in addition to the histogram data of the image. 

###Sliding Window Search

####1. Implementation

The sliding window specifies the image area where feature is extracted. The feature extraction method computes feature vectors based on the image data bounded by the sliding windows.

For training images being 64x64, a fixed window of the same size is used to train model. As to video frames, the window position is carefully chosen to map so that the window positions precisely matches the block positions in the HOG feature matrix. The code is implemented at `FeatureExtractor_WindowPlanner.py`.

The window moves at a step of 1/8 of window size, based on the HOG block step of 8 pixels and window size of 64 pixels. The window slides horizontally and vertically within a region of interest. The bigger the window, the lower in the image the region is. The windows scaling is implemented by zoom in or out the image. The window size and its corresponding region of interest is summerized in the following table.

| Window Size     | Proportion of Region of Interest from Image Bottom   |
| --------------- |:----------------------------------------------------:|
| 64x64           | 0.52 ~ 0.7                                           |
| 96x96           | 0.52 ~ 0.85                                          |
| 128x128         | 0.71 ~ 1                                             |


The following image shows an example of the size-varied random sliding windows and the searched result.

![alt text][image3]

####2. Examples of test images after optimized pipeline

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. Sliding windows varis the sizes and scanning on different regions of interest over the image. All the hits are overlayed on each other to generate a heatmap. The heatmap is optimized by tunning the hyper-parameter of the threhold. In particular, a smaller window generates more hits while the bigger window has fewer hits because it has less room to move around in the image. Therefore, varied weights are applied to the heatmap to further optimize the recognition perforamce. Here are some example images:

![example 1][image4_pipeline_eg1]
![example 2][image4_pipeline_eg2]
![example 3][image4_pipeline_eg3]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Click the following image to view the tracking demo on YouTube:

[![Vehicle-Tracking Video](https://www.youtube.com/embed/XRLHp-QBhCE/0.jpg)](https://www.youtube.com/embed/XRLHp-QBhCE "Vehicle Tracking Video on YouTube")


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Further Work

####1. Use HOG on a window-cropped image instead of the speedy HOG mapping method

To speed up feature extration, a HOG mapping method is used. Instead of apply HOG on each sub-image, a full video frame is fed to HOG. This give a HOG result matrix. Then we map the 64x64 window to the HOG matrix. 
However, there is difference between indivisual sub-image HOG result and the HOG map result. Each sub-image has a hard cut-off on the edge of its image, while the full image HOG offers gradience information between windows. 
The HOG mapping method is only an approximation to sub-image approximation. The classifier training uses the 64x64 data, so the HOG mapping method on the video mframe is essentially not consisstent with the training inputs to the classifier. As the result, although it improves the speed, the accuracy suffers. 
I tested on cropped fifty 64x64 images from a video frame. It give zero false positive. On the other hand, the full image scan makes 6~10 false positive predictions. 
As a fater computer can be used to speed up the process, sub-image HOG method can be used in the future development to improve accuracy.

