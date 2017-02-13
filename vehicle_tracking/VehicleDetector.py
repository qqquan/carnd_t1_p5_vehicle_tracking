import logging
logger = logging.getLogger(__name__)
logger.info('VehicleDetector module loaded')


from TrainingDataset import TrainingDataset
from FeatureExtractor import FeatureExtractor
from Classifier import Classifier
from HeatMap import FilteredHeatMap
from CarBoxList import CarBoxList

import cv2
import os
import pickle
import numpy as np
from scipy.ndimage.measurements import label




class VehicleDetector():



    def __init__(self, car_path='data/vehicles/', noncar_path = 'data/non-vehicles/', enable_checkpoint=False , 
                orient=9, pix_per_cell=8, cell_per_block=2,  spatial_shape=(16, 16), hist_bins=16, 
                filter_maxcount=5, heat_threhold=12, ):

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
                logger.debug('VehicleDetector: initialize a new checkpoint.')
                self.vehicle_classifier = self.trainClassifier(x_loc_list, y)

                with open('veh_classifier_checkpoint.pickle', 'wb') as handle:
                    pickle.dump(self.vehicle_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            self.vehicle_classifier = self.trainClassifier(x_loc_list, y)

        self.filtered_heat = FilteredHeatMap(max_count=filter_maxcount, threshold=heat_threhold)
        self.car_box_list = CarBoxList()

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

        heatmap = np.zeros_like(img_bgr[:,:,0])

        for a_win in detected_windows:

            ul_pos = a_win[0]
            br_pos = a_win[1]

            ul_row, ul_col = ul_pos
            br_row, br_col = br_pos

            heatmap[ul_row:br_row, ul_col:br_col] +=1

        self.filtered_heat.update(heatmap)

        return detected_windows, heatmap

    def drawBoxes(self, img_bgr, windows):

        bgr = np.copy(img_bgr)
        for a_win in windows:

            ul_pos = a_win[0]
            br_pos = a_win[1]

            ul_y, ul_x = ul_pos
            br_y, br_x = br_pos
            # logger.debug('window position: {}'.format(a_win))
            # cv2.rectangle(bgr, a_win[0], a_win[1],  (0,0,255))
            cv2.rectangle(bgr, (ul_x, ul_y), (br_x, br_y),  (255,0,0), thickness=1)

        return bgr 

    def labelCars(self, img_bgr):
        self.scanImg(img_bgr)
        labels = label( self.filtered_heat.getFilteredHeatmap() )

        labled_img = self.draw_labeled_bboxes(np.copy(img_bgr), labels)

        return labled_img, labels[0]  # return labeled image and label map

    def hightlightCars(self, img_bgr):
        return self.labelCars( img_bgr)


    def resetHeatmap(self):
        self.filtered_heat.reset()

    # from Ryan Keenan
    def draw_labeled_bboxes(self, img, labels):
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()

            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            self.car_box_list.update(car_number-1, bbox)


        car_boxes = self.car_box_list.getBoxList()

        for box in car_boxes:
            ul_pos = box[0]
            br_pos = box[1]

            ul_row = int(ul_pos[0])
            ul_col = int(ul_pos[1])

            br_row = int(br_pos[0])
            br_col = int(br_pos[1])

            cv2.rectangle(img, (ul_row, ul_col), (br_row, br_col), (255,0,0),6)

        return img
def main():
    from Util_Debug import visualize
    import glob

    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('######################### VehicleDetector - Module Test ############################')


    print('\n######################### Module Test ############################\n')



    car_detector = VehicleDetector(enable_checkpoint=True, heat_threhold=15)

    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_bgr = cv2.imread('data/test_images/test3.jpg')

    detected_windows, heatmap1 = car_detector.scanImg(video_img_bgr)
    print('Test frame 1: Number of detected windows: ', len(detected_windows))
    logger.info('Number of detected windows: {}.'.format(len(detected_windows)) )

    img_bgr_marked = car_detector.drawBoxes(video_img_bgr, detected_windows)



    video_img_bgr2 = cv2.imread('data/test_images/test6.jpg')

    detected_windows, heatmap2 = car_detector.scanImg(video_img_bgr2)
    print('Test frame 2: Number of detected windows: ', len(detected_windows))
    logger.info('Number of detected windows: {}.'.format(len(detected_windows)) )
    img_bgr_marked2 = car_detector.drawBoxes(video_img_bgr2, detected_windows)



    img_rgb_marked= cv2.cvtColor(img_bgr_marked, cv2.COLOR_BGR2RGB)
    img_rgb_marked2= cv2.cvtColor(img_bgr_marked2, cv2.COLOR_BGR2RGB)
    # video_img_rgb= cv2.cvtColor(video_img_bgr, cv2.COLOR_BGR2RGB)
    visualize(  [[img_rgb_marked, heatmap1], [img_rgb_marked2, heatmap2]], 
                [[ 'Marked Image - Example 1', 'Heatmap - Example 1'], ['Marked Image - Example 2', 'Heatmap - Example 2']],
                'data/outputs/car_detection_windows.png', enable_show=False)


    print('--------- Test label() ------ ')
    video_img_bgr1 = cv2.imread('data/test_images/test3.jpg')

    img_labled_bgr1, label_map1 = car_detector.labelCars(video_img_bgr1)

    video_img_bgr2 = cv2.imread('data/test_images/test6.jpg')
    car_detector.resetHeatmap() # image is not related to previous one
    img_labled_bgr2, label_map2 = car_detector.labelCars(video_img_bgr2)

    img_labled_rgb1= cv2.cvtColor(img_labled_bgr1, cv2.COLOR_BGR2RGB)
    img_labled_rgb2= cv2.cvtColor(img_labled_bgr2, cv2.COLOR_BGR2RGB)
    # video_img_rgb= cv2.cvtColor(video_img_bgr, cv2.COLOR_BGR2RGB)
    visualize(  [[img_labled_rgb1, label_map1], [img_labled_rgb2, label_map2]], 
                [[ ' Example 1 - Labeled Image', 'Example 1 - Label Map'], ['Example 2 - Labeled Image ', 'Example 2 - Label Map ']],
                'data/outputs/car_detection_labels.png', enable_show=False)


    print('--------- Test label() and filtering on continuous video frames ------ ')
    car_detector.resetHeatmap() # image is not related to previous one

    sample_dir='data/test_images/stream/'
    images_loc = glob.glob(sample_dir+'*.jpg')

    img_list =[]
    title_list = []
    for img_loc in images_loc:
        img_bgr = cv2.imread(img_loc)
        img_labled_bgr, label_map = car_detector.hightlightCars(img_bgr)
        img_labled_rgb = cv2.cvtColor(img_labled_bgr, cv2.COLOR_BGR2RGB)

        img_list.append([img_labled_rgb, label_map])
        img_filename = img_loc[-8:]
        print('Loading {}...'.format(img_filename))
        title_list.append( [img_filename+' - Labeled Image ', img_filename+' - Label Map '] )

        loc_to_save = sample_dir + 'labeled/labeled_' +img_filename
        visualize(img_list, title_list, loc_to_save)
        img_list =[]
        title_list = []
    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))