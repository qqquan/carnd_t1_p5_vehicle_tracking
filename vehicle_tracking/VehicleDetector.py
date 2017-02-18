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


    #TODO: Issue with spatial_shape=(32, 32), hist_bins=32,
    #remember to remove pickle file everytime the tuning parameter change, because training needs to be redone with new numbers.
    def __init__(self, car_path='data/vehicles/', noncar_path = 'data/non-vehicles/', enable_checkpoint=False , 
                feat_color_conv=cv2.COLOR_BGR2YCrCb, 
                orient=9, pix_per_cell=8, cell_per_block=2,  spatial_shape=(32, 32), hist_bins=32, 
                filter_maxcount=5, 
                heat_threhold=32, 
                win_scale_list= [1,             1.5,            2,        ], 
                ROI_list=       [(0.52,0.7),    (0.52,0.85),    (0.72,1),  ],

                ):

        dataset = TrainingDataset(car_path, noncar_path)
        x_loc_list = dataset.getXLoc()
        y = dataset.getY()

        example_img = cv2.imread(x_loc_list[0]) 
        self.training_image_shape = example_img.shape[:2]
        self.feature_extractor = FeatureExtractor(  training_image_shape=self.training_image_shape, 
                                                    color_conv=feat_color_conv,
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

        logger.debug('VehicleDetector - Classifier Accuracy: {}'.format(self.vehicle_classifier.getAccuracy()) )

        self.filtered_heat = FilteredHeatMap(max_count=filter_maxcount, threshold=heat_threhold)
        self.car_box_list = CarBoxList()

        self.win_scale_list = win_scale_list
        self.roi_list = ROI_list
        self.pix_per_cell= pix_per_cell

    def trainClassifier(self, x_loc_list, y):

        X = []
        for x_loc in x_loc_list:
            img_bgr = cv2.imread(x_loc)

            features,_ = self.feature_extractor.extractFeaturesAndWindows(img_bgr, win_scale=1)

            assert len(features) == 1
            X.extend(features) 

        classifier = Classifier(X,y)

        return classifier

    def scanImg(self, img_bgr, win_scale=None, 
                
                ):
        

        heatmap = np.zeros_like(img_bgr[:,:,0]).astype(np.float32)
        total_detected_windows = []

        if win_scale != None:
            # load normal single window size data
            win_scale_list = [win_scale]
            ROI_list = [(0.52, 1.0)]
        else:
            win_scale_list = self.win_scale_list 
            ROI_list = self.roi_list 

        for w_scale, roi in zip (win_scale_list, ROI_list):

            # logger.debug('scanImg(): start feature extraction')

            features, windows = self.feature_extractor.extractFeaturesAndWindows(np.copy(img_bgr), win_scale=w_scale, region_of_interest_row_ratio=roi)

            # logger.debug('scanImg(): start predicting. \n window scale: {}, features shape: {}, number of windows: {}'.format(w_scale, len(features), len(windows)))
            # logger.debug(' feature shape: {}, window shape: {}'.format((features[0].shape), (windows[0].shape)))
            predictions = self.vehicle_classifier.predict(features)
            # logger.debug('scanImg(): collect detected windows')

            detected_windows = [win for (win, pred) in zip(windows, predictions) if (pred==1)]
            # logger.debug('VehicleDetector - scanImg(): number of  detected windows for a scale: {}'.format(len(detected_windows)))        
            total_detected_windows.extend(detected_windows)
            for a_win in detected_windows:

                ul_pos = a_win[0]
                br_pos = a_win[1]

                ul_row, ul_col = ul_pos
                br_row, br_col = br_pos

                heatmap[ul_row:br_row, ul_col:br_col] += (1.0*w_scale*self.training_image_shape[0]/self.pix_per_cell)  #even out the number of windows and hits. smaller window has more test results, so it has higher weight on over all result


        self.filtered_heat.update(heatmap)

        return total_detected_windows, heatmap

    def drawBoxes(self, img_bgr, windows):

        bgr = np.copy(img_bgr)
        for a_win in windows:

            ul_pos = a_win[0]
            br_pos = a_win[1]

            ul_y, ul_x = ul_pos
            br_y, br_x = br_pos
            # logger.debug('window position: {}'.format(a_win))
            # cv2.rectangle(bgr, a_win[0], a_win[1],  (0,0,255))
            cv2.rectangle(bgr, (int(ul_x), int(ul_y)), (int(br_x), int(br_y)),  (255,0,0), thickness=1)

        return bgr 

    def labelCars(self, img_bgr):
        self.scanImg(img_bgr)
        labels = label( self.filtered_heat.getFilteredHeatmap() )

        labled_img = self.draw_labeled_bboxes(np.copy(img_bgr), labels)

        return labled_img, labels[0]  # return labeled image and label map

    def hightlightCars(self, img_bgr):

        img_highted, _ = self.labelCars( img_bgr)
        return img_highted


    def resetHeatmap(self):
        self.filtered_heat.reset()

    # from Ryan Keenan
    def draw_labeled_bboxes(self, img, labels):

        new_box_list = []
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()

            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bbox = ( (np.min(nonzeroy), np.min(nonzerox)), (np.max(nonzeroy), np.max(nonzerox)) ) # (row, col)

            new_box_list.append(np.array(bbox))

        self.car_box_list.update(new_box_list)
        car_boxes = self.car_box_list.getBoxList()

        # logger.debug('VehicleDetector - draw_labeled_bboxes(): number of cars detected: {}'.format(len(car_boxes)))
        # logger.debug('VehicleDetector - draw_labeled_bboxes(): detected car positions: {}'.format(new_box_list))
        # logger.debug('VehicleDetector - draw_labeled_bboxes(): filtered car positions: {}'.format(car_boxes))

        for box in car_boxes:
            ul_pos = box[0]
            br_pos = box[1]

            ul_row = int(ul_pos[0])
            ul_col = int(ul_pos[1])

            br_row = int(br_pos[0])
            br_col = int(br_pos[1])

            ul_x, ul_y = ul_col, ul_row
            br_x, br_y = br_col, br_row

            # logger.debug('VehicleDetector - draw_labeled_bboxes(): drawn car positions: {}'.format(( (ul_x, ul_y), (br_x, br_y))))

            cv2.rectangle(img, (ul_x, ul_y), (br_x, br_y), (255,0,0),6)

        return img


def main():
    from Util_Debug import visualize
    import glob
    import os.path

    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('######################### VehicleDetector - Module Test ############################')


    print('\n######################### Module Test ############################\n')



    car_detector = VehicleDetector(enable_checkpoint=True)

    print('\n\n######################### Video Frame Test ############################ \n')

    print('--------- Test multi-size windows ------ ')
    
    video_img_bgr = cv2.imread('data/test_images/697.jpg')



    detected_windows, heatmap1 = car_detector.scanImg(video_img_bgr)
    print('Test frame 1: Number of detected windows: ', len(detected_windows))
    logger.info('Number of detected windows: {}.'.format(len(detected_windows)) )

    img_bgr_marked = car_detector.drawBoxes(video_img_bgr, detected_windows)
    img_rgb_marked= cv2.cvtColor(img_bgr_marked, cv2.COLOR_BGR2RGB)

    visualize(  [[img_rgb_marked, heatmap1]], 
                [[ 'Marked Image  ' , 'Heatmap']],
                'data/outputs/car_detection_windows_multi_sizes.png', enable_show=True)

    labels = label( car_detector.filtered_heat.getFilteredHeatmap() )

    labled_img = car_detector.draw_labeled_bboxes(np.copy(video_img_bgr), labels)

    labled_img_rgb= cv2.cvtColor(labled_img, cv2.COLOR_BGR2RGB)
    loc_to_save = 'data/outputs/car_detection_windows_multi_sizes_labled.png'
    visualize(  [[labled_img_rgb, labels[0]]], 
                [[ 'Labled Image', 'Lable Map']],
                loc_to_save, enable_show=True)



    print('--------- Test windows  sizes and heatmap------ ')

    
    video_img_bgr = cv2.imread('data/test_images/1049.jpg')

    win_scale1 = 1

    car_detector.resetHeatmap() 
    detected_windows, heatmap1 = car_detector.scanImg(video_img_bgr, win_scale1)
    print('Test frame 1: Number of detected windows: ', len(detected_windows))
    logger.info('Number of detected windows: {}.'.format(len(detected_windows)) )

    img_bgr_marked = car_detector.drawBoxes(video_img_bgr, detected_windows)

    car_detector.resetHeatmap() 


    video_img_bgr2 = cv2.imread('data/test_images/1049.jpg')
    win_scale2 = 2
    detected_windows, heatmap2 = car_detector.scanImg(video_img_bgr2, win_scale=win_scale2)
    print('Test frame 2: Number of detected windows: ', len(detected_windows))
    logger.info('Number of detected windows: {}.'.format(len(detected_windows)) )
    img_bgr_marked2 = car_detector.drawBoxes(video_img_bgr2, detected_windows)



    img_rgb_marked= cv2.cvtColor(img_bgr_marked, cv2.COLOR_BGR2RGB)
    img_rgb_marked2= cv2.cvtColor(img_bgr_marked2, cv2.COLOR_BGR2RGB)
    # video_img_rgb= cv2.cvtColor(video_img_bgr, cv2.COLOR_BGR2RGB)
    visualize(  [[img_rgb_marked, heatmap1], [img_rgb_marked2, heatmap2]], 
                [[ 'Marked Image - Window Scale: '+str(win_scale1), 'Heatmap'], ['Marked Image - Window Scale: '+str(win_scale2), 'Heatmap']],
                'data/outputs/car_detection_windows.png', enable_show=True)



    


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



    print('--------- Test scanImg() and Heatmaps on all test images ------ ')
    logger.info('--------- Test scanImg() and Heatmaps on all test images ------ ')

    sample_dir='data/test_images/'
    images_loc = glob.glob(sample_dir+'*.jpg')

    img_list =[]
    title_list = []
    for img_loc in images_loc:

        car_detector.resetHeatmap() # image is not related to previous one

        video_img_bgr = cv2.imread(img_loc)

        path, img_filename = os.path.split(img_loc)   



        detected_windows, heatmap1 = car_detector.scanImg(video_img_bgr)
        print(img_filename+': Number of detected windows: ', len(detected_windows))
        logger.debug('Testing: '+ img_filename)
        logger.info(img_filename+': Number of detected windows: {}.'.format(len(detected_windows)) )

        img_bgr_marked = car_detector.drawBoxes(video_img_bgr, detected_windows)





        img_rgb_marked= cv2.cvtColor(img_bgr_marked, cv2.COLOR_BGR2RGB)
        loc_to_save = sample_dir + 'test_result/heated_' +img_filename
        visualize(  [[img_rgb_marked, heatmap1]], 
                    [[ img_filename+ ': Marked Image', 'Heatmap']],
                    loc_to_save, enable_show=False)


        labels = label( car_detector.filtered_heat.getFilteredHeatmap() )

        labled_img = car_detector.draw_labeled_bboxes(np.copy(video_img_bgr), labels)

        labled_img_rgb= cv2.cvtColor(labled_img, cv2.COLOR_BGR2RGB)
        loc_to_save = sample_dir + 'test_result/labled_' +img_filename
        visualize(  [[labled_img_rgb, labels[0]]], 
                    [[ img_filename+ ': Labled Image', 'Lable Map']],
                    loc_to_save, enable_show=False)


    print('--------- Test label() and filtering on continuous video frames ------ ')
    car_detector.resetHeatmap() # image is not related to previous one

    sample_dir='data/test_images/stream/'
    images_loc = glob.glob(sample_dir+'*.jpg')

    img_list =[]
    title_list = []
    for img_loc in images_loc:
        img_bgr = cv2.imread(img_loc)
        img_labled_bgr, label_map = car_detector.labelCars(img_bgr)
        img_labled_rgb = cv2.cvtColor(img_labled_bgr, cv2.COLOR_BGR2RGB)

        img_list.append([img_labled_rgb, label_map])
        # img_filename = img_loc[-8:]
        path, img_filename = os.path.split(img_loc)   

        print('Loading {}...'.format(img_filename))
        logger.info('Tested {}.'.format(img_filename))
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