import logging
logger = logging.getLogger(__name__)
logger.info('VehicleDetector module loaded')


from TrainingDataset import TrainingDataset
from FeatureExtractor import FeatureExtractor
from Classifier import Classifier

import cv2
import os
import pickle
import numpy as np

class VehicleDetector():



    def __init__(self, car_path='data/vehicles/', noncar_path = 'data/non-vehicles/', enable_checkpoint=False , 
                orient=9, pix_per_cell=8, cell_per_block=2,  spatial_shape=(16, 16), hist_bins=32 ):

        dataset = TrainingDataset(car_path, noncar_path)
        x_loc_list = dataset.getXLoc()
        y = dataset.getY()

        example_img = cv2.imread(x_loc_list[0]) 
        self.feature_extractor = FeatureExtractor(  training_image_shape=example_img.shape[:2], 
                                                    orient=orient, 
                                                    pix_per_cell=pix_per_cell,
                                                    cell_per_block=cell_per_block,
                                                    spatial_shape=spatial_shape,
                                                    hist_bins=hist_bins)


        if enable_checkpoint: 
            #load checkpoint data
            if os.path.isfile('veh_classifier_checkpoint.pickle') :
                logger.debug('VehicleDetector: load classifier checkpoint.')
                with open('veh_classifier_checkpoint.pickle', 'rb') as handle:
                    self.vehicle_classifier = pickle.load(handle)
            else: 
                self.vehicle_classifier = self.trainClassifier(x_loc_list, y)
                logger.debug('VehicleDetector: initialize a new checkpoint.')

                with open('veh_classifier_checkpoint.pickle', 'wb') as handle:
                    pickle.dump(self.vehicle_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            self.vehicle_classifier = self.trainClassifier(x_loc_list, y)



    def trainClassifier(self, x_loc_list, y):

        X = []
        for x_loc in x_loc_list:
            img_bgr = cv2.imread(x_loc)

            features,_ = self.feature_extractor.extractFeaturesAndWindows(img_bgr)

            assert len(features) == 1
            X.extend(features) 

        classifier = Classifier(X,y)

        return classifier

    def scanImg(self, img_bgr):
        

        features, windows = self.feature_extractor.extractFeaturesAndWindows(img_bgr)

        predictions = self.vehicle_classifier.predict(features)

        detected_windows = [win for (win, pred) in zip(windows, predictions) if (pred==1)]

        return detected_windows

    def drawBoxes(self, img_bgr, windows):

        bgr = np.copy(img_bgr)
        for a_win in windows:

            ul_pos = a_win[0]
            br_pos = a_win[1]

            ul_y, ul_x = ul_pos
            br_y, br_x = br_pos
            # logger.debug('window position: {}'.format(a_win))
            # cv2.rectangle(bgr, a_win[0], a_win[1],  (0,0,255))
            cv2.rectangle(bgr, (ul_x, ul_y), (br_x, br_y),  (255,0,0), thickness=3)

        return bgr 


def main():

    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('######################### VehicleDetector - Module Test ############################')

    from Util_Debug import visualize

    print('\n######################### Module Test ############################\n')



    car_detector = VehicleDetector(enable_checkpoint=True)

    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_bgr = cv2.imread('data/test_images/test6.jpg')

    detected_window = car_detector.scanImg(video_img_bgr)
    print('Number of detected windows: ', len(detected_window))

    img_bgr_marked = car_detector.drawBoxes(video_img_bgr, detected_window)



    video_img_bgr2 = cv2.imread('data/test_images/test3.jpg')

    detected_window = car_detector.scanImg(video_img_bgr2)
    print('Number of detected windows: ', len(detected_window))

    img_bgr_marked2 = car_detector.drawBoxes(video_img_bgr2, detected_window)



    img_rgb_marked= cv2.cvtColor(img_bgr_marked, cv2.COLOR_BGR2RGB)
    img_rgb_marked2= cv2.cvtColor(img_bgr_marked2, cv2.COLOR_BGR2RGB)
    # video_img_rgb= cv2.cvtColor(video_img_bgr, cv2.COLOR_BGR2RGB)
    visualize([[img_rgb_marked2], [img_rgb_marked]],[[ 'Marked Image - Example 1'], ['Marked Image - Example 2']], enable_show=True)
    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))