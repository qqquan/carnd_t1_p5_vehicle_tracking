import logging
logger = logging.getLogger(__name__)
logger.info('FeatureExtractor submodule loaded')


from FeatureExtractor_Hog import HogExtractor
from FeatureExtractor_Color import ColorExtractor
from FeatureExtractor_WindowPlanner import WindowPlanner

import numpy as np
import cv2



class FeatureExtractor():

    def __init__(self, training_image_shape=(64,64), orient=9, pix_per_cell=8, cell_per_block=2,  spatial_shape=(16, 16), hist_bins=32):

        self.window_planner = WindowPlanner(training_image_shape, pix_per_cell = pix_per_cell)
        self.hog_extractor = HogExtractor(orient=orient, pix_per_cell = pix_per_cell, cell_per_block=cell_per_block)
        self.color_extractor = ColorExtractor(spatial_shape=spatial_shape, hist_bins=hist_bins)

        self.training_image_shape=training_image_shape

    def extractFeaturesAndWindows(self, img, win_scale=1, region_of_interest_row_ratio=(0.55, 1)):
        """Extract feature vectors and corresponding inspected windows for an image.
        
        Args:
            img (TYPE): Description
        
        Returns:
            TYPE: feature vectors and corresponding inspected windows
        """


        #//TODO: add window zooming with different ROI

        if np.max(img) > 1:

            img = img.astype(np.float32)/255


        img_scaled = cv2.resize(img, ( int(img.shape[1]/win_scale), int(img.shape[0]/win_scale) ))

        logger.debug('FeatureExtractor -  scale = {}'.format(win_scale))
        logger.debug('FeatureExtractor -  new shape = {}, old shape ={}'.format(img_scaled.shape, img.shape))

        windows = self.window_planner.getHogWindows(img_scaled, region_of_interest_row_ratio ) # windows of upper-left and bottom-right pixel positions of hog blocks
        scaled_windows = np.multiply(windows, win_scale)


        color_features = self.color_extractor.getFeatures(img_scaled, windows)

        hog_features = self.hog_extractor.getFeatures(img_scaled, windows)


        color_feature_len = len(color_features[0])
        hog_feature_len = len(hog_features[0])

        features = []
        for color_per_win, hog_per_win in zip(color_features, hog_features):
            feat = np.concatenate((color_per_win, hog_per_win))
            features.append(feat)


        #TODO: Make a new window list if image is scaled

        return features, scaled_windows

def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    import cv2


    #####################################################
    # Training Images
    #####################################################
    logger.info(' ####  FeatureExtractor - Training Image Test  ###')
    logger.info('Training image Test ---------- \n')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    feature_extractor = FeatureExtractor(training_image_shape=training_img_brg.shape[:2], pix_per_cell = 8, cell_per_block = 2)

    features, windows = feature_extractor.extractFeaturesAndWindows(training_img_brg)
    print('Number of windows: ', len(windows))
    print('Number of features: ', len(features))
    print('Feature shape for the first window: ', features[0].shape)
    assert(len(features) == len(windows))
    training_feature_len = len(features[0])
    print('traning feature size: ', training_feature_len)

    print('\n\n######################### Video Frame Test ############################ \n')
    logger.info('Video Frame Test -------------- \n')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')

    frame_features, frame_windows = feature_extractor.extractFeaturesAndWindows(video_img_brg, win_scale=1.5)
    print('number of features: ', len(frame_features))
    print('video frame feature size: ', len(frame_features[0]))

    count = 0
    for feat in frame_features:
        assert len(feat)==training_feature_len, 'Expect frame feature has the same size as training feature. Idx: {}. video frame feature len: {}, training feature length: {}'.format(count, len(feat), training_feature_len)
        count+=1

    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))