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
[image3]: ./doc/car_detection_windows_multi_sizes_resized.png
[image4_pipeline_eg1]: ./doc/heated_test1_resized.jpg
[image4_pipeline_eg2]: ./doc/heated_test3_resized.jpg
[image4_pipeline_eg3]: ./doc/heated_test4_resized.jpg

[image7]: ./doc/output_bboxes_new.png
[video1]: ./project_video.mp4


---
###Design

The top layer abstraction is implemented at`VehicleDetector.py`, while `VehicleDetection_Main.py` is the application code that generates the video output. A centralized feature extraction module is designed at `FeatureExtractor.py`, which uniformly processes both training images and video frames. 

The following diagram shows the design of software architecture and logic flow.

![alt text][SDD_image0]

###Histogram of Oriented Gradients (HOG)

####1. Extract HOG features from the training images.

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

In general, pixels_per_cell of 8 captures the key car features, such as lamps and windows, in proportion to the 64x64 training data. The color space YCrCb isolates out the illumination effects, so lighting variation can be suppressed in the model. The other parameters more or less provide a lot more data or details of the feature to be fed for model training. The trade-off is that smaller feature vector speeds up the video processing, while a bigger vector improves model prediction accuracy.

####3. Classifier

I trained a linear SVM using HOG and color features. In order to speed up the process, HOG vector is extracted over the whole video frame and then mapped to each small window. The color feature is extracted from direct pixel values of a size-reduced sub-image, in addition to the histogram data of the image. Before fitting the model, StandardScaler is used to normalize the data. The code is at `Classifier.py`.

###Sliding Window Search

####1. Implementation

The sliding window specifies the image area where feature is extracted. The feature extraction method computes feature vectors based on the image data bounded by the sliding windows.

For training images being 64x64, a fixed window of the same size is used to train model. As to video frames, the window position is carefully chosen to map so that the window positions precisely matches the block positions in the HOG feature matrix. The code is implemented at `FeatureExtractor_WindowPlanner.py`.

The window moves at a step of 1/8 of window size, based on the HOG block step of 8 pixels and window size of 64 pixels. The window slides horizontally and vertically within a region of interest. The smaller the window, the higher in the image the region is to capture distant cars. The windows scaling is implemented by zoom in or out the image. The window size and its corresponding region of interest is summarized in the following table.

| Window Size     | Proportion of Region of Interest from Image Bottom   |
| --------------- |:----------------------------------------------------:|
| 64x64           | 0.52 ~ 0.7                                           |
| 96x96           | 0.52 ~ 0.85                                          |
| 128x128         | 0.71 ~ 1                                             |


The following image shows an example of the size-varied random sliding windows and the searched result.

![alt text][image3]

####2. Examples of test images after optimized pipeline

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. Sliding windows varies the sizes and scanning on different regions of interest over the image. All the hits are overlaid on each other to generate a heatmap. The heatmap is optimized by tunning the hyper-parameter of the threshold. In particular, a smaller window generates more hits while the bigger window has fewer hits because it has less room to move around in the image. Therefore, varied weights are applied to the heatmap to further optimize the recognition performance. Here are some example images:

![example 1][image4_pipeline_eg1]
![example 2][image4_pipeline_eg2]
![example 3][image4_pipeline_eg3]

---

### Video Implementation

####1. Final Video Output 
Click the following image to view the tracking demo on YouTube:

[![Vehicle-Tracking Video](https://github.com/qqquan/carnd_t1_p5_vehicle_tracking/raw/master/doc/youtube_video1.png)](https://www.youtube.com/embed/XRLHp-QBhCE "Vehicle Tracking Video on YouTube")


####2. Filter for false positives and overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  `doOverlap()` function is implemented at `CarBoxList.py` to remove overlapping boxes with limited filtering capability. 

Here's an example result showing a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image7]



---

###Future Work

####1. Use HOG on a window-cropped image instead of the speedy HOG mapping method

To speed up feature extraction, a HOG mapping method is used. Instead of apply HOG on each sub-image, a full video frame is fed to HOG. This gives a HOG result matrix. Then we map the 64x64 window to the HOG matrix. 
However, there is difference between individual sub-image HOG result and the HOG map result. Each sub-image has a hard cut-off on the edge of its image, while the full image HOG offers gradience information between windows. 
The classifier training uses the 64x64 data, so the HOG mapping method on the video frame is essentially not consistent with the training inputs to the classifier. As the result, although it improves the speed, the accuracy suffers. 
I tested on cropped fifty 64x64 images from a video frame. It give zero false positive. On the other hand, the full image scan makes 6~10 false positive predictions. 
As a faster computer can be used to speed up the process, sub-image HOG method can be used in the future development to improve accuracy.

####2. Incoming Cars
The incoming cars are not filtered out, because it is useful surrounding information. Further work can be done to add a switch to turn off incoming traffic detection based on region-of interest, lane-detection, moving direction of bounding boxes, etc. 

####3. Bounding box filtering and Object Tracking
The output video shows the current filter is still not smooth and misses at some nesting cases. Further work is needed to have a full-fledged Car class to track and filter each car's position, speed, distance, etc.

####4. Classifiers and Dataset
This work uses LinearSVC for its speediness in prototyping and near 99% accuracy with the Udacity dataset. There are many other good classifier to evaluate such as, RBF SVM. Deep Learning methods are another excellent resource for more challenging driving situations and greater training dataset. Because of the well-encapsulated code, new classifiers or models and be swapped and tested at `Classifier.py`.