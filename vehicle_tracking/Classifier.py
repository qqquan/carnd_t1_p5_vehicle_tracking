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


def main():
    import numpy as np
    from TrainingDataset import TrainingDataset
    import cv2

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



    debug_num = 100 
    print('\n\n*********\nLimited number of images for testing: {}! \n*********'.format(2*debug_num))

    #####################################################
    # Classifier Training 
    #####################################################
    print('\n\n######################### Module Test on Classifier Training ############################ \n')
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

    vehicle_classifier = Classifier(X,test_y)

    print('Accuracy: ', vehicle_classifier.getAccuracy())

    assert vehicle_classifier.getAccuracy() > 0.5


    print('\n\n######################### Module Test on Classifier Prediction ############################ \n')
    print('---------- Test predict() on Car images ------------')
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
    video_img_brg = cv2.imread('data/test_images/test6.jpg')

    features, windows = feature_extractor.extractFeaturesAndWindows(video_img_brg)

    print(' total features number: {}'.format(len(features)))
    print(' type of feature vectors X: {}'.format(type(features)))
    print(' size of a feature : {}'.format( len(features[0]) ))
    print(' shape of a feature : {}'.format( (features[0].shape) ))

    predictions = []
    for feat in features:

        pred = vehicle_classifier.predict(feat)
        predictions.append(pred) 
    print('Non-car Prediction hits = {}, misses = {}'.format(pred_hit_num, len(predicitons)-pred_hit_num))
    print('Non-car Prediction Accuracy: ', pred_hit_num/len(predicitons) )
    assert pred_hit_num/len(predicitons) > 0.5
    
    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))