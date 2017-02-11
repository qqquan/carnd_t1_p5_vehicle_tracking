import cv2
import numpy as np

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features




class ColorExtractor():


    def __init__(self, spatial_shape=(16, 16), hist_bins=32):
        self.spatial_shape = spatial_shape
        self.hist_bins = hist_bins

    def getFeature(self, img, windows):
        features = []
        for window in windows:

            ul_pos = window[0]
            dr_pos = window[1]

            ul_row, ul_col = ul_pos
            dr_row, dr_col = dr_pos

            subimg = img[ul_row:dr_row, ul_col:dr_col]

            spatial_features = bin_spatial(subimg, size=self.spatial_shape)
            hist_features = color_hist(subimg, nbins=self.hist_bins)

            features.append(np.concatenate((spatial_features, hist_features)))

        return features

def main():
    color_extractor = ColorExtractor()

    #####################################################
    # Training Images
    #####################################################
    print('\n######################### Training Image Test ############################\n')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    img = training_img_brg
    print('Training Image shape: ', img.shape)

    w1 = ((0,0),img.shape[:2])
    windows = [w1]

    features = color_extractor.getFeature(img, windows)

    print('Number of features: ', len(features))
    assert len(features)==1

    feat0_shape = features[0].shape
    print('Feature shape for the 1st window: ', feat0_shape)
    assert feat0_shape[0]>0


    #####################################################
    # Video Frame
    #####################################################
    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')
    img = video_img_brg
    print('Video frame shape: ', img.shape)


    w1 = ((0,0),img.shape[:2])
    w2 = ((400,600),(464, 664))
    w3 = ((500,0),(564, 64))

    windows = [w1, w2, w3]
    features = color_extractor.getFeature(img, windows)

    print('Number of features: ', len(features))
    assert len(features)==3

    feat0_shape = features[0].shape
    print('Feature shape for the 1st window: ', feat0_shape)
    assert feat0_shape[0]>0



if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))