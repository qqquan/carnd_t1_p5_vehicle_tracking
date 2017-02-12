import logging
logger = logging.getLogger(__name__)
logger.info('WindowPlanner submodule loaded')

import cv2


# based on Udacity course material
def generateSlidingWindows(img_shape, rowcol_stride_pix, col_start_stop=[None, None], row_start_stop=[None, None] , rowcol_window=(64, 64)):
    # If col and/or row start/stop positions not defined, set to image size
    if col_start_stop[0] is None or col_start_stop[1] is None:
        col_start = 0
        col_stop = img_shape[1]
    else:
        col_start = col_start_stop[0]
        col_stop = col_start_stop[1]     

    if row_start_stop[0] is None or row_start_stop[1] is None:
        row_start = 0
        row_stop = img_shape[0]
    else:
        row_start = row_start_stop[0]
        row_stop = row_start_stop[1]     

    # Compute the span of the region to be searched
    # Compute the number of pixels per step in col/row
    col_step = rowcol_stride_pix[1]
    row_step = rowcol_stride_pix[0]

    col_range = range(col_start, col_stop+1, col_step)
    row_range = range(row_start, row_stop+1, row_step)

    # Compute the number of windows in col/row
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding col and row window positions

    for col in col_range:
        for row in row_range:
            # Calculate each window position
            upper_pos = (row, col)
            lower_pos = (row+rowcol_window[0], col+rowcol_window[1])
            box = (upper_pos, lower_pos)
            # Append window position to list
            window_list.append(box)
    # Return the list of windows

    return window_list


class WindowPlanner():
    def __init__(self, training_image_shape = (64,64), pix_per_cell = 8, cell_per_block = 2):

        assert len(training_image_shape) == 2, 'Expect a width x length 2D shape, but training_image_shape = {} \n'. format(training_image_shape) 
        self.training_image_shape = training_image_shape

        pix_per_blk_pos = pix_per_cell

        self.pix_step = 8 


    # window size has to equal to training_image_shape
    def getHogWindows(self, img):
        """eatch window corner is a hog cell corner. as Hog block moves by cell
        
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

            #  sliding step has to be divisable by pix_per_blk_pos or pix_per_cell


            windows.append([])

        return windows


def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(' #### WindowPlanner -  Module Testing  ###')




    #####################################################
    # Training Images
    #####################################################
    print('\n######################### Training Image Test ############################\n')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    windows_planner = WindowPlanner(training_img_brg.shape[:2])

    windows = windows_planner.getHogWindows(training_img_brg)

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

    windows = windows_planner.getHogWindows(video_img_brg)

    print('Number of windows: ', len(windows))
    # assert(len(windows) == 1)

    print('Shape of the first window :', windows[0])

    #####################################################
    # Module Test on generateSlidingWindows()
    #####################################################
    print('\n\n######################### Module Test on generateSlidingWindows() ############################ \n')
    print('------- square window test------- ')
    img_col = 1280
    img_row = 720
    col_start = 0
    col_stop = 1280
    row_start = 400
    row_stop = 720
    pix_per_cell = 8
    win_size = 64
    windows = generateSlidingWindows(   img_shape=(img_row, img_col), 
                                        rowcol_stride_pix=(pix_per_cell,pix_per_cell), 
                                        col_start_stop=[col_start, col_stop], 
                                        row_start_stop=[row_start, row_stop] , 
                                        rowcol_window=(win_size, win_size))

    w0 = windows[0]
    w1 = windows[1]

    w0_row = w0[1][0] - w0[0][0]
    w0_col = w0[1][1] - w0[0][1]
    w1_row = w1[1][0] - w1[0][0]
    w1_col = w1[1][1] - w1[0][1]

    print('window 0: ({},{}),  ({},{})'.format(w0[0][0], w0[0][1] ,w0[1][0]  ,w0[1][1]  ))
    print('window 1: ({},{}),  ({},{})'.format(w1[0][0], w1[0][1] ,w1[1][0]  ,w1[1][1]  ))
    print('window rows: {}, {}'.format(w0_row, w1_row))
    print('window columns: {}, {}'.format(w0_col, w1_col))

    assert w0_row == win_size
    assert w0_col == win_size
    assert w1_row == win_size
    assert w1_col == win_size

    row_step = w1[0][0] - w0[0][0]
    col_step = w1[0][1] - w0[0][1]
    print('row_step = ', row_step)
    print('col_step = ', col_step)
    assert (row_step+col_step) == pix_per_cell, "window moves a step either in column or row axis, not both"

    print('\n------- rectangular window test------- ')
    img_col = 1280
    img_row = 720
    col_start = 0
    col_stop = 1280
    row_start = 400
    row_stop = 720

    row_step = 8
    col_step = 16
    win_size = (100,200)
    windows = generateSlidingWindows(   img_shape=(img_row, img_col), 
                                        rowcol_stride_pix=(row_step,col_step), 
                                        col_start_stop=[col_start, col_stop], 
                                        row_start_stop=[row_start, row_stop] , 
                                        rowcol_window=(win_size[0], win_size[1]))

    w0 = windows[0]
    w1 = windows[1]

    w0_row = w0[1][0] - w0[0][0]
    w0_col = w0[1][1] - w0[0][1]
    w1_row = w1[1][0] - w1[0][0]
    w1_col = w1[1][1] - w1[0][1]

    print('window 0: ({},{}),  ({},{})'.format(w0[0][0], w0[0][1] ,w0[1][0]  ,w0[1][1]  ))
    print('window 1: ({},{}),  ({},{})'.format(w1[0][0], w1[0][1] ,w1[1][0]  ,w1[1][1]  ))
    print('window rows: {}, {}'.format(w0_row, w1_row))
    print('window columns: {}, {}'.format(w0_col, w1_col))

    assert w0_row == win_size[0]
    assert w0_col == win_size[1]
    assert w1_row == win_size[0]
    assert w1_col == win_size[1]

    w01_row_step = w1[0][0] - w0[0][0]
    w01_col_step = w1[0][1] - w0[0][1]
    print('row_step = ', w01_row_step)
    print('col_step = ', w01_col_step)
    assert (w01_row_step == row_step ) or (w01_col_step==col_step), "window moves a step either in column or row axis, not both"

    print('\n------- rectangular window test - all windows covered ------- ')
    is_init = True
    w0 = None
    for w in windows:
        if is_init:
            is_init = False
            w0 = w # w0 stores the last window
            break

        w1 = w


        w0_row = w0[1][0] - w0[0][0]
        w0_col = w0[1][1] - w0[0][1]
        w1_row = w1[1][0] - w1[0][0]
        w1_col = w1[1][1] - w1[0][1]
        assert w0_row == win_size[0], 'past window - row_size = {}, win_size = {}'.format( w0_row,   win_size[0])
        assert w0_col == win_size[1], 'past window - col_size = {}, win_size = {}'.format( w0_col,   win_size[1])
        assert w1_row == win_size[0], 'new window - row_size = {}, win_size = {}'.format( w1_row,   win_size[0])
        assert w1_col == win_size[1], 'new window - col_size = {}, win_size = {}'.format( w1_col,   win_size[1])
        w01_row_step = w1[0][0] - w0[0][0]
        w01_col_step = w1[0][1] - w0[0][1]

        assert (w01_row_step == row_step ) or (w01_col_step==col_step), "window moves a step either in column or row axis, not both"
        assert ((w01_row_step == row_step ) and (w01_col_step==col_step) ) == False, "Expected: window moves a step either in column or row axis. \
                                                                                        Actual:  the window moves at both direction. \
                                                                                        w01_row_step = {},  w01_col_step = {}".format(w01_row_step, w01_col_step)

        w0 = w1 # w0 stores the last window




if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
        