import logging
logger = logging.getLogger(__name__)
logger.info('FeatureExtractor submodule loaded')


from FeatureExtractor_Hog import HogExtractor
from FeatureExtractor_Color import ColorExtractor
from FeatureExtractor_WindowPlanner import WindowPlanner

import numpy as np

class FeatureExtractor():

    def __init__(self, training_image_shape=(64,64), pix_per_cell = 8, cell_per_block = 2):

        self.window_planner = WindowPlanner(training_image_shape, pix_per_cell = 8)
        self.hog_extractor = HogExtractor(pix_per_cell = 8, cell_per_block = 2)
        self.color_extractor = ColorExtractor()


    def extractFeatureAndWindows(self, img):
        print('//TODO: add window zooming with different ROI')
        if np.max(img) > 1:
            # jpg image ranges 0~255. png file does not need this because its range is 0~1
            img = img.astype(np.float32)/255

        windows = self.window_planner.getHogWindows(img) # windows of upper-left and bottom-right pixel positions of hog blocks
        logger.debug('FeatureExtractor - window_planner.getHogWindows  - number of windows: {}'.format(len(windows)))

        color_features = self.color_extractor.getFeatures(img, windows)

        hog_features = self.hog_extractor.getFeatures(img, windows)

        features=[]
        for color_per_win, hog_per_win in zip(color_features, hog_features):
            feat = np.concatenate((color_per_win, hog_per_win))
            features.append(feat)

        return features, windows

def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    import cv2

    feature_extractor = FeatureExtractor()

    #####################################################
    # Training Images
    #####################################################
    logger.info(' ####  FeatureExtractor - Training Image Test  ###')

    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    features, windows = feature_extractor.extractFeatureAndWindows(training_img_brg)
    print('Number of windows: ', len(windows))
    print('Number of features: ', len(features))
    print('Feature shape for the first window: ', features[0].shape)
    assert(len(features) == len(windows))



    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))