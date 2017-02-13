import logging
logger = logging.getLogger(__name__)
logger.info('VehicleDetector module loaded')


from TrainingDataset import TrainingDataset
from FeatureExtractor import FeatureExtractor
from Classifier import Classifier

import cv2
import os
import pickle


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
        
        logger.debug('VehicleDetector - scanImg() - total window number: {}'.format(len(windows)))
        logger.debug('VehicleDetector - scanImg() - total features number: {}'.format(len(features)))
        logger.debug('VehicleDetector - scanImg() - type of a feature : {}'.format(type(features[0])))
        logger.debug('VehicleDetector - scanImg() - size of a feature : {}'.format( len(features[0]) ))
        logger.debug('VehicleDetector - scanImg() - shape of a feature : {}'.format( (features[0].shape) ))

        predictions = self.vehicle_classifier.predict(features)

        detected_windows = [win for (win, pred) in zip(windows, predictions) if (pred==1)]

        logger.debug('total detected number: {}'.format(len(detected_windows)))

        return detected_windows


def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('######################### VehicleDetector - Module Test ############################')
    print('\n######################### Module Test ############################\n')



    car_detector = VehicleDetector(enable_checkpoint=True)

    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')

    detected_window = car_detector.scanImg(video_img_brg)
    print('Number of detected windows: ', len(detected_window))

    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))