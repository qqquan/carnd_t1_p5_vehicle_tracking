import logging
logger = logging.getLogger(__name__)
logger.info('Classifier module loaded')


from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

class Classifier():

    def __init__(self, X, y):
        

        self.svc = LinearSVC()

        self.X_scaler = StandardScaler().fit(X)

        scaled_X = self.X_scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

        self.svc.fit(X_train, y_train)

        self.accuracy = round(self.svc.score(X_test, y_test), 4)


    def getAccuracy(self):
        return self.accuracy

    def predict(self, X):
        scaled_X = self.X_scaler.transform(X)
        return self.svc.predict(scaled_X)




def main(    debug_num=100, use_pre_trained_classifier=True ):

    




    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('######################### VehicleDetector - Module Test ############################')

    import numpy as np
    from TrainingDataset import TrainingDataset
    import cv2
    import glob
    import os.path

    dataset = TrainingDataset('data/vehicles/', 'data/non-vehicles/')

    x_loc_list = dataset.getXLoc()
    y = dataset.getY()
    print('Number of images: ', len(x_loc_list))
    print('Number of labels: ', len(y))

    print('Number of vehicle images: ', len(dataset.getVehicleImgList()))
    print('Number of vehicle lables: ', np.count_nonzero(dataset.getY()))
    assert(len(dataset.getVehicleImgList()) == np.count_nonzero(dataset.getY()))

    num_of_non_vehicle_img = len(dataset.getNonVehicleImgList())
    num_of_non_vehicle_labels = len(dataset.getY()) - np.count_nonzero(dataset.getY())
    print('Number of non-vehicle images: ', num_of_non_vehicle_img)
    print('Number of non-vehicle lables: ', num_of_non_vehicle_labels)
    assert(num_of_non_vehicle_labels == num_of_non_vehicle_img)   



    print('\n\n*********\nLimited number of images for testing: {}! \n*********'.format(2*debug_num))

    #####################################################
    # Classifier Training 
    #####################################################
    print('\n\n######################### Module Test on Classifier Training ############################ \n')
    logger.info('\n\n######################### Module Test on Classifier Training ############################ \n')
    test_x_loc_list = x_loc_list[-debug_num:-1]  + x_loc_list[0:debug_num]
    test_y = np.concatenate((y[-debug_num:-1] ,y[0:debug_num]))

    assert len(test_x_loc_list)>0
    assert len(test_x_loc_list) == len(test_y)

    from FeatureExtractor import FeatureExtractor

    feature_extractor = FeatureExtractor()
    X = []
    for x_loc in test_x_loc_list:
        img_bgr = cv2.imread(x_loc)

        features,_ = feature_extractor.extractFeaturesAndWindows(img_bgr)

        assert len(features) == 1
        X.extend(features)

    assert len(X) == len(test_y), 'Num of feature vectors: {}, number of labels: {}'.format(len(X), len(y) )

    if use_pre_trained_classifier == False:
        vehicle_classifier = Classifier(X,test_y)
    else:
        import os.path
        import pickle
        #load checkpoint data
        if os.path.isfile('veh_classifier_checkpoint.pickle') :
            logger.debug('VehicleDetector: load classifier checkpoint.')
            with open('veh_classifier_checkpoint.pickle', 'rb') as handle:
                vehicle_classifier = pickle.load(handle)
        else: 
            X = []
            for x_loc in x_loc_list:
                img_bgr = cv2.imread(x_loc)

                features,_ = feature_extractor.extractFeaturesAndWindows(img_bgr)

                assert len(features) == 1
                X.extend(features)
            vehicle_classifier = Classifier(X,y)



    print('Accuracy: ', vehicle_classifier.getAccuracy())

    assert vehicle_classifier.getAccuracy() > 0.5





    print('\n\n######################### Module Test on Classifier Prediction ############################ \n')
    print('---------- Test predict() on Car images ------------')
    logger.info('---------- Test predict() on Car images ------------')
    assert debug_num*2 < len(dataset.getXLoc())/2

    car_loc_list =x_loc_list[debug_num:2*debug_num]

    assert len(car_loc_list)>0

    X = []
    for x_loc in car_loc_list:
        img_bgr = cv2.imread(x_loc)

        features,_ = feature_extractor.extractFeaturesAndWindows(img_bgr)

        assert len(features) == 1
        X.extend(features)



    predicitons = vehicle_classifier.predict(X)
    pred_hit_num = np.count_nonzero(predicitons)
    print('Car Prediction hits = {}, misses = {}'.format(pred_hit_num, len(predicitons)-pred_hit_num))
    print('Car Prediction Accuracy: ', pred_hit_num/len(predicitons) )
    assert pred_hit_num/len(predicitons) > 0.5

    print('\n---------- Test predict() on Non-car images ------------')
    logger.info('\n---------- Test predict() on Non-car images ------------')
    noncar_loc_list =x_loc_list[-2*debug_num:-1*debug_num]


    assert len(noncar_loc_list)>0

    X = []
    for x_loc in noncar_loc_list:
        img_bgr = cv2.imread(x_loc)

        features,_ = feature_extractor.extractFeaturesAndWindows(img_bgr)

        assert len(features) == 1
        X.extend(features)

    print(' total features number: {}'.format(len(X)))
    print(' type of feature vectors X: {}'.format(type(X)))
    print(' type of a feature : {}'.format(type(X[0])))
    print(' size of a feature : {}'.format( len(X[0]) ))
    print(' shape of a feature : {}'.format( (X[0].shape) ))

    predicitons = vehicle_classifier.predict(X)
    print(' type of predicitons : {}'.format(type(predicitons)))
    print(' type of a prediciton : {}'.format(type(predicitons[0])))
    pred_hit_num = len(predicitons) -np.count_nonzero(predicitons)
    print('Non-car Prediction hits = {}, misses = {}'.format(pred_hit_num, len(predicitons)-pred_hit_num))
    print('Non-car Prediction Accuracy: ', pred_hit_num/len(predicitons) )
    assert pred_hit_num/len(predicitons) > 0.5

    print('\n\n######################### Video Frame Test ############################ \n')
    logger.info('\n\n######################### Video Frame Test ############################ \n')
    video_img_brg = cv2.imread('data/test_images/537.jpg')

    features, windows = feature_extractor.extractFeaturesAndWindows(video_img_brg,win_scale=2)

    print(' total features number: {}'.format(len(features)))
    logger.info(' total features number: {}'.format(len(features)))
    print(' type of feature vectors X: {}'.format(type(features)))
    print(' size of a feature : {}'.format( len(features[0]) ))
    print(' shape of a feature : {}'.format( (features[0].shape) ))
    logger.info(' shape of a feature : {}'.format( (features[0].shape) ))
    print(' shape of a reshaped feature : {}'.format( features[0].reshape(1,-1).shape) )    
    # predictions = []
    # for feat in features:
    #     #TODO: (feat.reshape(1,-1))? OR JUST feat?
    #     pred = vehicle_classifier.predict(feat)  
    #     predictions.append(pred) 
    predictions = vehicle_classifier.predict(features) 
    pred_hit_num = len(predicitons) -np.count_nonzero(predicitons)
    logger.info('Number of predictions: {}'.format(len(predictions)))
    print('Video Frame Prediction hits = {}, misses = {}'.format(pred_hit_num, len(predictions)-pred_hit_num))
    logger.info('Video Frame Prediction hits = {}, misses = {}'.format(pred_hit_num, len(predictions)-pred_hit_num))
    print('Video Frame Prediction Accuracy: ', pred_hit_num/len(predictions) )
    # assert pred_hit_num/len(predictions) > 0.5
   

    detected_windows = [win for (win, pred) in zip(windows, predictions) if (pred==1)]

    img_bgr_marked = drawBoxes(video_img_brg, detected_windows)
    DBG_saveAllWindowedImages(video_img_brg, detected_windows)
    cv2.imshow('Marked Video Frame', img_bgr_marked)
    cv2.waitKey()



    print('\n\n######################### 64x64 window-cropped Video Frames: Classifier Prediction ############################ \n')

    logger.info('---------- 64x64 window-cropped Video Frames: Classifier Prediction ------------')


    sample_dir='data/test_images/croped_537_subimg/'
    car_loc_list = glob.glob(sample_dir+'*.jpg')

    assert len(car_loc_list)>0

    X = []
    for x_loc in car_loc_list:
        img_bgr = cv2.imread(x_loc)
        path, img_filename = os.path.split(x_loc)   

        feature,_ = feature_extractor.extractFeaturesAndWindows(img_bgr)
        predi = vehicle_classifier.predict(feature)

        assert len(feature) == 1
        X.extend(feature)

        logger.debug( img_filename+ ' prediction:  ' + str(predi))
        print( img_filename+ ' prediction:  ' + str(predi))



    predicitons = vehicle_classifier.predict(X)
    pred_hit_num = np.count_nonzero(predicitons)
    print('Car Prediction hits = {}, misses = {}'.format(pred_hit_num, len(predicitons)-pred_hit_num))
    print('Car Prediction Accuracy: ', pred_hit_num/len(predicitons) )
    # assert pred_hit_num/len(predicitons) > 0.5


    print('\n**************** All Tests Passed! *******************')

def drawBoxes(img_bgr, windows):
    import numpy as np
    import cv2
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




DBG_counter=0
import cv2
def DBG_saveAllWindowedImages(img_bgr, windows):
    global DBG_counter

    # img_bgr = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite('data/debug_all_images/window_cropped_video_frames/video_frame.jpg', img_bgr)


    for win in windows:
        ul_pos = win[0]
        dr_pos = win[1]

        ul_row, ul_col = ul_pos
        dr_row, dr_col = dr_pos

        subimg = img_bgr[ul_row:dr_row, ul_col:dr_col]
        subimg_scaled = cv2.resize(subimg, ( 64, 64) )

        cv2.imwrite('data/debug_all_images/window_cropped_video_frames/'+ str(DBG_counter)+'.png', subimg_scaled)
        DBG_counter += 1





if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))


