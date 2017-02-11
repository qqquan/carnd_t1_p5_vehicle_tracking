from FeatureExtractor_Hog import HogExtractor
from FeatureExtractor_Color import ColorExtractor
from FeatureExtractor_WindowPlanner import WindowPlanner

class FeatureExtractor():

    def __init__(self):
        self.window_planner = WindowPlanner()
        self.hog_extractor = HogExtractor()
        self.color_extractor = ColorExtractor()

    def extractFeatureAndWindows(img):

        if np.max(img) > 1:
            # jpg image ranges 0~255. png file does not need this because its range is 0~1
            img = img.astype(np.float32)/255

        windows = self.window_planner.getWindows(img) # windows of upper-left and bottom-right pixel positions

        hog_feature = self.hog_extractor(img, windows)

        color_feature = self.color_extractor(img, windows)

        feature = np.concatenate((hog_feature, color_feature))

        return feature, windows

def main():
    import cv2

    feature_extractor = FeatureExtractor()

    img_brg = cv2.imread('data/test_images/test6.jpg')

    features, windows = feature_extractor.extractFeatureAndWindows(img_brg)
    print('Feature shape: ', features.shape)
    print('Number of windows: ', len(windows))
    assert(len(features) == len(windows))

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))