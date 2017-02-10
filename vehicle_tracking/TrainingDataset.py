
import numpy as np
import glob


#TODO: matplotlib image will read these in on a scale of 0 to 1, but cv2.imread() will scale them from 0 to 255.

#TODO: Be sure to normalize your training data. Use sklearn.preprocessing.StandardScaler() to normalize your feature vectors for training your classifier. Then apply the same scaling to each of the feature vectors you extract from windows in your test images.

#TODO: Random shuffling of data. When dealing with image data that was extracted from video, you may be dealing with sequences of images where your target object (vehicles in this case) appear almost identical in a whole series of images. In such a case, even a randomized train-test split will be subject to overfitting because images in the training set may be nearly identical to images in the test set. For the project vehicles dataset, the GTI* folders contain time-series data. In the KITTI folder, you may see the same vehicle appear more than once, but typically under significantly different lighting/angle from other instances.


class TrainingDataset():

    def __init__(self, vehicle_data_path, nonvehicle_data_path):

        self.ls_x_loc_vehicle = self.loadImgLoc(vehicle_data_path)
        self.ls_x_loc_nonvehicle = self.loadImgLoc(nonvehicle_data_path)


    def loadImgLoc(self, path):
        ls_img_loc = glob.glob(path+'**/*.png', recursive=True)
        return ls_img_loc

    def getVehicleImgList(self):
        return self.ls_x_loc_vehicle

    def getNonVehicleImgList(self):
        return self.ls_x_loc_nonvehicle

    def getXLocList(self):
        ls_x_loc =  self.ls_x_loc_vehicle + self.ls_x_loc_nonvehicle
        return ls_x_loc

    def getY(self):
        np_y = np.hstack((np.ones(len(self.ls_x_loc_vehicle)), np.zeros(len(self.ls_x_loc_nonvehicle))))
        return np_y


def main():
    dataset = TrainingDataset('data/vehicles/', 'data/non-vehicles/')

    print('Number of images: ', len(dataset.getXLocList()))
    print('Number of labels: ', len(dataset.getY()))

    print('Number of vehicle images: ', len(dataset.getVehicleImgList()))
    print('Number of vehicle lables: ', np.count_nonzero(dataset.getY()))
    assert(len(dataset.getVehicleImgList()) == np.count_nonzero(dataset.getY()))

    num_of_non_vehicle_img = len(dataset.getNonVehicleImgList())
    num_of_non_vehicle_labels = len(dataset.getY()) - np.count_nonzero(dataset.getY())
    print('Number of non-vehicle images: ', num_of_non_vehicle_img)
    print('Number of non-vehicle lables: ', num_of_non_vehicle_labels)
    assert(num_of_non_vehicle_labels == num_of_non_vehicle_img)   


if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))