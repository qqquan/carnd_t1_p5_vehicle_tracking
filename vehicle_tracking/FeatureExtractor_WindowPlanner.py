
import cv2


class WindowPlanner(object):
    """docstring for WindowPlanner"""
    def __init__(self, training_image_shape = (64,64)):
        super(WindowPlanner, self).__init__()

        assert len(training_image_shape) == 2, 'Expect a width x length 2D shape, but training_image_shape = {} \n'. format(training_image_shape) 
        self.training_image_shape = training_image_shape
        self.windows = {}

    def getWindows(self, img):
        """Summary
        
        Args:
            img (TYPE): Description
        
        Returns:
            LIST:   A list of window position in tuple. 
                    Each position contains two tuples: upper left corner of (row, column) and bottom right corner of (row, column))
        """
        windows = []

        training_shape = self.training_image_shape

        if (img.shape[0] <= training_shape[0]) and (img.shape[1] <= training_shape[1]):
            if img.shape != training_shape:
                img = cv2.resize(img, training_shape)
            windows=   [  ((0,0), training_shape) ] # scan-window is over the whole image
        else:
            print('//TODO: Find scan windows for a video frame')
            windows.append([])

        return windows

def main():




    #####################################################
    # Training Images
    #####################################################
    print('\n######################### Training Image Test ############################\n')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    windows_planner = WindowPlanner(training_img_brg.shape[:2])

    windows = windows_planner.getWindows(training_img_brg)

    print('Number of windows: ', len(windows))
    assert(len(windows) == 1)

    print('Shape of the first window :', windows[0])
    assert(windows[0][0] == (0,0)) # upper left pos should be at 0
    assert(windows[0][1] == training_img_brg.shape[:2]) #bottom right pos

    #####################################################
    # Video Frame
    #####################################################
    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')

    print('Video frame shape: ', video_img_brg.shape)

    windows = windows_planner.getWindows(video_img_brg)

    print('Number of windows: ', len(windows))
    # assert(len(windows) == 1)

    print('Shape of the first window :', windows[0])

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
        