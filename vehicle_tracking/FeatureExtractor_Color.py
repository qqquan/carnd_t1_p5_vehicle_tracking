import cv2
import numpy as np

# Define a function to compute binned color features  
# from Udacity course materials
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# # from Udacity course materials
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

    def getFeatures(self, img, windows):
        """crop image based on windows, then calculate the feature vector for each cropped image.
        
        Args:
            img (numpy array): Description
            windows (LIST): Description
        
        Returns:
            LIST:  a list of feature vectors. Each feature is extracted from a cropped image off a window.
        """
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

    features = color_extractor.getFeatures(img, windows)

    print('Number of feature vectors: ', len(features))
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


    window_size = 64
    w1 = ((0,0),(window_size,window_size))
    w2 = ((400,600),(400+window_size, 600+window_size))
    w3 = ((500,0),(500+window_size, 0+window_size))

    windows = [w1, w2, w3]
    features = color_extractor.getFeatures(img, windows)

    print('Number of feature vectors: ', len(features))
    assert len(features)==3

    feat0_shape = features[0].shape
    print('Feature shape for the 1st window: ', feat0_shape)
    assert feat0_shape[0]>0

    feat1_shape = features[1].shape
    print('Feature shape for the 2nd window: ', feat1_shape)
    assert feat1_shape[0]>0


if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))