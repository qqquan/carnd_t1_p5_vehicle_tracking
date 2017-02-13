import logging
logger = logging.getLogger(__name__)
logger.info('FeatureExtractor submodule loaded')


from FeatureExtractor_Hog import HogExtractor
from FeatureExtractor_Color import ColorExtractor
from FeatureExtractor_WindowPlanner import WindowPlanner

import numpy as np

class FeatureExtractor():

    def __init__(self, training_image_shape=(64,64), orient=9, pix_per_cell=8, cell_per_block=2,  spatial_shape=(16, 16), hist_bins=32):

        self.window_planner = WindowPlanner(training_image_shape, pix_per_cell = pix_per_cell)
        self.hog_extractor = HogExtractor(orient=orient, pix_per_cell = pix_per_cell, cell_per_block=cell_per_block)
        self.color_extractor = ColorExtractor(spatial_shape=spatial_shape, hist_bins=hist_bins)


    def extractFeaturesAndWindows(self, img):
        """Extract feature vectors and corresponding inspected windows for an image.
        
        Args:
            img (TYPE): Description
        
        Returns:
            TYPE: feature vectors and corresponding inspected windows
        """


        #//TODO: add window zooming with different ROI
        if np.max(img) > 1:
            # jpg image ranges 0~255. png file does not need this because its range is 0~1
            img = img.astype(np.float32)/255

        windows = self.window_planner.getHogWindows(img) # windows of upper-left and bottom-right pixel positions of hog blocks
        # logger.debug('FeatureExtractor - window_planner.getHogWindows  - number of windows: {}'.format(len(windows)))

        color_features = self.color_extractor.getFeatures(img, windows)

        hog_features = self.hog_extractor.getFeatures(img, windows)

        features=[]

        color_feature_len = len(color_features[0])
        hog_feature_len = len(hog_features[0])
        logger.debug('FeatureExtractor - Expected Color Feature Length: {}'.format(color_feature_len))
        logger.debug('FeatureExtractor - Expected Hog Feature Length: {}'.format(hog_feature_len))
        for color_per_win, hog_per_win in zip(color_features, hog_features):
            feat = np.concatenate((color_per_win, hog_per_win))
            features.append(feat)

            # if len(color_per_win) != color_feature_len:
            #     logger.error('FeatureExtractor -Expected color feature len: {}, Actual len: {}'.format(color_feature_len, len(color_per_win) ))
            # if len(hog_per_win) != hog_feature_len:
            #     logger.error('FeatureExtractor - Expected hog feature len: {}, Actual len: {}'.format(hog_feature_len, len(hog_per_win) ))

        #TODO: Make a new window list if image is scaled

        return features, windows

def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    import cv2


    #####################################################
    # Training Images
    #####################################################
    logger.info(' ####  FeatureExtractor - Training Image Test  ###')

    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    feature_extractor = FeatureExtractor(training_image_shape=training_img_brg.shape[:2], pix_per_cell = 8, cell_per_block = 2)

    features, windows = feature_extractor.extractFeaturesAndWindows(training_img_brg)
    print('Number of windows: ', len(windows))
    print('Number of features: ', len(features))
    print('Feature shape for the first window: ', features[0].shape)
    assert(len(features) == len(windows))
    training_feature_len = len(features)

    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')

    frame_features, frame_windows = feature_extractor.extractFeaturesAndWindows(video_img_brg)

    frame_feature_len = len(frame_features[0])
    count = 0
    for feat in frame_features:
        assert len(feat)==frame_feature_len, 'Idx: {}. feat len: {}, frame_feature_len: {}'.format(count, len(feat), frame_feature_len)
        count+=1

    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))