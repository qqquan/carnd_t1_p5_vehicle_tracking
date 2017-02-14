import logging
logger = logging.getLogger(__name__)
logger.info('WindowPlanner submodule loaded')

import cv2


# based on Udacity course material
def generateSlidingWindows(img_shape, rowcol_stride_pix, col_start_stop=[None, None], row_start_stop=[None, None] , rowcol_window=(64, 64)):
    # If col and/or row start/stop positions not defined, set to image size

    if row_start_stop[0] is None or row_start_stop[1] is None:
        row_start = 0
        row_stop = img_shape[0]
    else:
        row_start = row_start_stop[0]
        row_stop = row_start_stop[1]    

    if col_start_stop[0] is None or col_start_stop[1] is None:
        col_start = 0
        col_stop = img_shape[1]
    else:
        col_start = col_start_stop[0]
        col_stop = col_start_stop[1]     

    assert row_start <= row_stop
    assert col_start <= col_stop
    assert row_stop <= img_shape[0]
    assert col_stop <= img_shape[1]
    
    assert rowcol_window[0] <= img_shape[0], 'Expect search window being smaller than image. rowcol_window[0] = {}, img_shape[0] = {}'.format(rowcol_window[0], img_shape[0])
    assert rowcol_window[1] <= img_shape[1], 'Expect search window being smaller than image. rowcol_window[0] = {}, img_shape[0] = {}'.format(rowcol_window[1], img_shape[1])

    # Compute the span of the region to be searched
    # Compute the number of pixels per step in col/row
    row_step = rowcol_stride_pix[0]
    col_step = rowcol_stride_pix[1]

    upperleft_row_range = range(row_start, row_stop+1-rowcol_window[0], row_step) # row position of upper left corners of windows
    upperleft_col_range = range(col_start, col_stop+1-rowcol_window[1], col_step)

    # Compute the number of windows in col/row
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding col and row window positions

    for col in upperleft_col_range:
        for row in upperleft_row_range:
            # Calculate each window position
            upper_pos = (row, col)
            lower_pos = (row+rowcol_window[0], col+rowcol_window[1])
            box = (upper_pos, lower_pos)
            # Append window position to list
            window_list.append(box)

    #TODO: if the the last window does not cover the row-col edges, make additional windows stepping from end edge for one step. If the window size is small, the missing edge pixels do not have a huge impact on detection area.

    return window_list


class WindowPlanner():
    def __init__(self, training_image_shape = (64,64), pix_per_cell = 8):

        assert len(training_image_shape) == 2, 'Expect a width x length 2D shape, but training_image_shape = {} \n'. format(training_image_shape) 
        self.training_image_shape = training_image_shape

        self.pix_step = pix_per_cell  # pixels of a block step

    # window size has to equal to training_image_shape
    def getHogWindows(self, img, region_of_interest_row_ratio=(0.55, 1) ):
        """eatch window corner is a hog cell corner. as Hog block moves by cell
        
        Args:
            img (TYPE): Description
        
        Returns:
            LIST:   A list of window position in tuple. 
                    Each position contains two tuples: upper left corner of (row, column) and bottom right corner of (row, column))
        """
        logger.debug('WindowPlanner - region_of_interest_row_ratio ={}'.format(region_of_interest_row_ratio))
        windows = []

        training_shape = self.training_image_shape

        if (img.shape[0] <= training_shape[0]) and (img.shape[1] <= training_shape[1]):
            if img.shape != training_shape:
                img = cv2.resize(img, training_shape)
            windows =  [  ((0,0), training_shape) ] # scan-window is over the whole image
        else:


            #  sliding step has to be divisible by pix of a block step or pix_per_cell
            ROI_min, RIO_max = region_of_interest_row_ratio

            row_start =  int(( ROI_min *img.shape[0] )//self.pix_step * self.pix_step) # make sure the starting position is at a hog block edge
            row_stop = int(RIO_max*img.shape[0])
            col_start = 0 
            col_stop = img.shape[1]
            windows = generateSlidingWindows(   img_shape=img.shape, 
                                                rowcol_stride_pix=(self.pix_step, self.pix_step ), 
                                                col_start_stop=[col_start, col_stop], 
                                                row_start_stop=[row_start, row_stop] , 
                                                rowcol_window=self.training_image_shape)

        return windows


def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(' #### WindowPlanner -  Module Testing  ###')





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



    print('img_col = ', img_col)
    print('img_row = ', img_row)
    print('col_start = ', col_start)
    print('col_stop = ', col_stop)
    print('row_start = ', row_start)
    print('row_stop = ', row_stop)
    print('pix_per_cell = ', pix_per_cell)

    print('win_size = ', win_size)


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
    print('Number of windows: ', len(windows))

    row_window_positions = ((row_stop-row_start - win_size)//pix_per_cell) +1
    col_window_positions = ((col_stop-col_start - win_size)//pix_per_cell) +1
    expected_win_num =  row_window_positions* col_window_positions
    assert len(windows) == expected_win_num, 'Expected window number = {}. Actual window number = {}. \n row_window_positions = {}, col_window_positions = {}.'.format(expected_win_num, len(windows), row_window_positions, col_window_positions)


    print('\n------- rectangular window test------- ')
    img_col = 1280
    img_row = 720
    col_start = 0
    col_stop = 1280
    row_start = 400
    row_stop = 720

    row_step = 8
    col_step = 16
    win_shape = (100,200)

    print('img_col = ', img_col)
    print('img_row = ', img_row)
    print('col_start = ', col_start)
    print('col_stop = ', col_stop)
    print('row_start = ', row_start)
    print('row_stop = ', row_stop)
    print('row_step = ', row_step)
    print('col_step = ', col_step)
    print('win_shape = ', win_shape)

    windows = generateSlidingWindows(   img_shape=(img_row, img_col), 
                                        rowcol_stride_pix=(row_step,col_step), 
                                        col_start_stop=[col_start, col_stop], 
                                        row_start_stop=[row_start, row_stop] , 
                                        rowcol_window=win_shape)
    print('Number of windows: ', len(windows))

    row_window_positions = ((row_stop-row_start - win_shape[0])//row_step) +1
    col_window_positions = ((col_stop-col_start - win_shape[1])//col_step) +1
    expected_win_num =  row_window_positions* col_window_positions
    assert len(windows) == expected_win_num, 'Expected window number = {}. Actual window number = {}. \n row_window_positions = {}, col_window_positions = {}.'.format(expected_win_num, len(windows), row_window_positions, col_window_positions)


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

    assert w0_row == win_shape[0], 'w0_row = {}, win_shape[0]= {} '.format( w0_row,win_shape[0])
    assert w0_col == win_shape[1], 'w0_col = {}, win_shape[1]= {} '.format( w0_col ,win_shape[1])
    assert w1_row == win_shape[0], 'w1_row = {}, win_shape[0]= {} '.format( w1_row ,win_shape[0])
    assert w1_col == win_shape[1], 'w1_col = {}, win_shape[1]= {} '.format( w1_col ,win_shape[1])

    w01_row_step = w1[0][0] - w0[0][0]
    w01_col_step = w1[0][1] - w0[0][1]
    print('row_step = ', w01_row_step)
    print('col_step = ', w01_col_step)
    assert (w01_row_step == row_step ) or (w01_col_step==col_step), "window moves a step either in column or row axis, not both"

    w_last = windows[-1]
    print('Last window position: ({},{}),  ({},{})'.format(w_last[0][0], w_last[0][1] ,w_last[1][0]  ,w_last[1][1]  ))

    print('\n------- rectangular window test - all windows covered ------- ')
    print('Number of windows: ', len(windows))

    row_window_positions = ((row_stop-row_start - win_shape[0])//row_step) +1
    col_window_positions = ((col_stop-col_start - win_shape[1])//col_step) +1
    expected_win_num =  row_window_positions* col_window_positions


    assert len(windows) == expected_win_num, 'Expected window number = {}. Actual window number = {}. \n row_window_positions = {}, col_window_positions = {}.'.format(expected_win_num, len(windows), row_window_positions, col_window_positions)

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
        assert w0_row == win_shape[0], 'past window - row_size = {}, win_shape = {}'.format( w0_row,   win_shape[0])
        assert w0_col == win_shape[1], 'past window - col_size = {}, win_shape = {}'.format( w0_col,   win_shape[1])
        assert w1_row == win_shape[0], 'new window - row_size = {}, win_shape = {}'.format( w1_row,   win_shape[0])
        assert w1_col == win_shape[1], 'new window - col_size = {}, win_shape = {}'.format( w1_col,   win_shape[1])
        w01_row_step = w1[0][0] - w0[0][0]
        w01_col_step = w1[0][1] - w0[0][1]

        assert (w01_row_step == row_step ) or (w01_col_step==col_step), "window moves a step either in column or row axis, not both"
        assert ((w01_row_step == row_step ) and (w01_col_step==col_step) ) == False, "Expected: The window moves a step either in column or row axis. \
                                                                                        Actual:  The window moves at both directions. \
                                                                                        w01_row_step = {},  w01_col_step = {}".format(w01_row_step, w01_col_step)

        w0 = w1 # w0 stores the last window


    #####################################################
    # Training Images
    #####################################################
    print('\n######################### Training Image Test ############################\n')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    pix_per_cell = 8
    training_image_shape = training_img_brg.shape[:2]

    windows_planner = WindowPlanner(training_image_shape=training_image_shape, pix_per_cell=pix_per_cell )

    windows = windows_planner.getHogWindows(training_img_brg)

    print('Number of windows: ', len(windows))
    assert len(windows) == 1, 'Expect only one window for a training image. '

    print('Shape of the first window :', windows[0])
    assert(windows[0][0] == (0,0)) # upper left pos should be at 0
    assert(windows[0][1] == training_img_brg.shape[:2]) #bottom right pos



    #####################################################
    # Video Frame
    #####################################################
    print('\n\n######################### Video Frame Test ############################ \n')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')

    ROI_min = 0.6
    ROI_max = 0.9 
    region_of_interest_row_ratio = (ROI_min, ROI_max)


    
    print('Video frame shape: ', video_img_brg.shape)
    img_row = video_img_brg.shape[0]
    img_col = video_img_brg.shape[1]

    row_step = pix_per_cell
    col_step = pix_per_cell
    win_shape = training_image_shape

    print('pix_per_cell = ', pix_per_cell)
    print('region_of_interest_row_ratio = ', region_of_interest_row_ratio)
    print('training_image_shape = ', training_image_shape)
    print('img_col = ', img_col)
    print('img_row = ', img_row)
    print('row_step = ', row_step)
    print('col_step = ', col_step)
    print('win_shape = ', win_shape)

    windows = windows_planner.getHogWindows(video_img_brg, region_of_interest_row_ratio)

    # assert(len(windows) == 1)

    print('The first window :', windows[0])
    print('The last window :', windows[-1])

    # test every window
    is_init = True
    w0 = None
    for w in windows:

        # test region of interest
        upper = w[0]
        lower = w[1]

        lower_row = lower[0]
        lower_col = lower[1]
        assert (lower_row >= (ROI_min*img_row) ) and (lower_row<=ROI_max*img_row) , 'lower_row = {}, min = {}, max = {} '.format(lower_row,  ROI_min*img_row  , ROI_max*img_row)


        if is_init:
            is_init = False
            w0 = w # w0 stores the last window
            break

        w1 = w


        w0_row = w0[1][0] - w0[0][0]
        w0_col = w0[1][1] - w0[0][1]
        w1_row = w1[1][0] - w1[0][0]
        w1_col = w1[1][1] - w1[0][1]

        #test window shapes are the same as training shape
        assert w0_row == win_shape[0], 'past window - row_size = {}, win_shape = {}'.format( w0_row,   win_shape[0])
        assert w0_col == win_shape[1], 'past window - col_size = {}, win_shape = {}'.format( w0_col,   win_shape[1])
        assert w1_row == win_shape[0], 'new window - row_size = {}, win_shape = {}'.format( w1_row,   win_shape[0])
        assert w1_col == win_shape[1], 'new window - col_size = {}, win_shape = {}'.format( w1_col,   win_shape[1])
        w01_row_step = w1[0][0] - w0[0][0]
        w01_col_step = w1[0][1] - w0[0][1]

        #test distance between windows
        assert (w01_row_step == row_step ) or (w01_col_step==col_step), "window moves a step either in column or row axis"
        assert ((w01_row_step == row_step ) and (w01_col_step==col_step) ) == False, "Expected: The window moves a step either in column or row axis. \
                                                                                        Actual:  The window moves at both directions. \
                                                                                        w01_row_step = {},  w01_col_step = {}".format(w01_row_step, w01_col_step)

        w0 = w1 # w0 stores the last window


    row_start = ROI_min*img_row
    row_stop = ROI_max*img_row
    print('row_start = ', row_start)
    print('row_stop = ', row_stop)
    col_start = 0
    col_stop = img_col
    row_window_positions = ((row_stop-row_start - win_shape[0])//pix_per_cell) +1
    col_window_positions = ((col_stop-col_start - win_shape[1])//pix_per_cell) +1
    expected_win_num =  row_window_positions* col_window_positions
    print('Number of windows: ', len(windows))
    assert len(windows) == expected_win_num, 'Expected window number = {}. Actual window number = {}. \n row_window_positions = {}, col_window_positions = {}.'.format(expected_win_num, len(windows), row_window_positions, col_window_positions)


    print('\n**************** All Tests Passed! *******************\n')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))
        