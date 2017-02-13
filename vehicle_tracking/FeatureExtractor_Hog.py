import logging
logger = logging.getLogger(__name__)
logger.info('HogExtractor submodule loaded')


import numpy as np
import cv2
from skimage.feature import hog


# Define a function to return HOG features and visualization
# from Udacity course materials
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features




class HogExtractor():

    def __init__(self, orient=9, pix_per_cell=8, cell_per_block=2):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block

    def getFeatures(self, img, windows):
        features = []

        ch1 = img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]

        # vector = the number of block positions * the number of cells per block * the number of orientations, e.g. 7*7*2*2*9
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        past_feat_len = 0
        for win in windows:
            ul_pos = win[0] # upper left position
            br_pos = win[1] # bottom right position

            ul_row, ul_col = ul_pos
            br_row, br_col = br_pos

            if ul_row % self.pix_per_cell:
                logger.error( 'The pixel position does not map to any hog matrix element. ul_row = {}, pix_per_cell = {}.'.format(ul_row, self.pix_per_cell)  )
            if ul_col % self.pix_per_cell:
                logger.error( 'The pixel position does not map to any hog matrix element. ul_col = {}, pix_per_cell = {}.'.format(ul_col, self.pix_per_cell) )
            if br_row % self.pix_per_cell:
                logger.error( 'The pixel position does not map to any hog matrix element. br_row = {}, pix_per_cell = {}.'.format(br_row, self.pix_per_cell) )
            if br_col % self.pix_per_cell:
                logger.error( 'The pixel position does not map to any hog matrix element. br_col = {}, pix_per_cell = {}.'.format(br_col, self.pix_per_cell) )



            # convert from pixel positions to block positions
            blk_pos_ul_row = ul_row//self.pix_per_cell
            blk_pos_ul_col = ul_col//self.pix_per_cell
            blk_pos_br_row = br_row//self.pix_per_cell-1 # the end block position is one less than the cell number!!
            blk_pos_br_col = br_col//self.pix_per_cell-1


            feat1 = hog1[blk_pos_ul_row:blk_pos_br_row, blk_pos_ul_col:blk_pos_br_col ].reshape(-1)
            feat2 = hog2[blk_pos_ul_row:blk_pos_br_row, blk_pos_ul_col:blk_pos_br_col ].reshape(-1)
            feat3 = hog3[blk_pos_ul_row:blk_pos_br_row, blk_pos_ul_col:blk_pos_br_col ].reshape(-1)

            feat = np.concatenate((feat1, feat2, feat3))

            features.append(feat)

            if past_feat_len ==0:
                past_feat_len = len(feat)
            else:
                if (len(feat) != past_feat_len):
                    logger.error('HogExtractor - Expected hog feature len: {}, Actual len: {}. \n \
                                 feat1 size: {}, feat2 size: {}, feat3 size: {}'\
                                 .format(past_feat_len, len(feat) , len(feat1) , len(feat2), len(feat3)   )
                                 )

                    logger.error('HogExtractor - Hog1 shape {}  '.format(hog1.shape))
                    logger.error('HogExtractor - Hog2 shape {}  '.format(hog2.shape))
                    logger.error('HogExtractor - Hog3 shape {}  '.format(hog3.shape))
                    logger.error('blk_pos_br_row - blk_pos_ul_row = Expected 7 of total blocks per a window, Actual: {} '.format(blk_pos_br_row - blk_pos_ul_row ))
                    logger.error('blk_pos_br_row: {}  '.format(blk_pos_br_row))
                    logger.error('blk_pos_ul_row: {}  '.format(blk_pos_ul_row))
                    logger.error('blk_pos_br_col - blk_pos_ul_col = Expected 7 of total blocks per a window, Actual: {} '.format(blk_pos_br_col - blk_pos_ul_col ))
                    logger.error('blk_pos_br_col: {}  '.format(blk_pos_br_col))
                    logger.error('blk_pos_ul_col: {}  '.format(blk_pos_ul_col))

                    logger.error('HogExtractor - Window Positions - Upper: {}. Lower: {}  '.format(ul_pos, br_pos))
                    past_feat_len=len(feat)

            # logger.debug('HogExtractor - Window Position - x - {}:{}.   y - {}:{} '.format(ul_col, br_col, ul_row, br_row))
            # logger.debug('HogExtractor - HOG Block Position - x - {}:{}.   y - {}:{} '.format(blk_pos_ul_col, blk_pos_br_col, blk_pos_ul_col, blk_pos_br_row))

        return features

def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    orient=9
    pix_per_cell=8
    cell_per_block=2

    hog_extractor = HogExtractor(orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

    #####################################################
    # Training Images
    #####################################################
    print('\n######################### Training Image Test ############################\n')
    logger.info(' HogExtractor Training Image Test')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    img = training_img_brg
    print('Training Image shape: ', img.shape)

    w1 = ((0,0),img.shape[:2])
    windows = [w1]

    # function under test:
    features = hog_extractor.getFeatures(img, windows)

    print('Number of feature vectors: ', len(features))
    assert len(features)==1

    feat0_shape = features[0].shape
    print('Feature shape for the 1st window: ', feat0_shape)
    assert feat0_shape[0]>0

    feat0_size = len(features[0])
    print('Feature size for the 1st window: ', feat0_size)
    window_side_len = img.shape[0]
    blk_pos_per_window = window_side_len//pix_per_cell -1
    num_hog_channels = 3 # assuming all color channels are fed to hog()
    assert img.shape[0]==img.shape[1], "training image is assumed a square shape"
    feature_num = blk_pos_per_window*blk_pos_per_window*cell_per_block*cell_per_block*orient *num_hog_channels

    print('window_side_len: ', window_side_len)
    print('blk_pos_per_window: ', blk_pos_per_window)
    print('cell_per_block: ', cell_per_block)
    print('orient: ', orient)
    print('num_hog_channels: ', num_hog_channels)
    print('Expected feature size: ', feature_num)
    assert feat0_size==(feature_num), 'Expected feature number: {}'.format(feature_num)

    training_feature_num = feat0_size


    #####################################################
    # Video Frame
    #####################################################
    print('\n\n######################### Video Frame Test ############################ \n')
    logger.info(' HogExtractor Video Frame Test')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')
    img = video_img_brg
    print('Video frame shape: ', img.shape)




    window_size = 64
    w1 = ((0,0),(window_size,window_size))
    w2 = ((400,600),(400+window_size, 600+window_size))
    w3 = ((504,0),(504+window_size, 0+window_size))
    windows = [w1, w2, w3]

    features = hog_extractor.getFeatures(img, windows)

    print('Number of feature vectors: ', len(features))
    assert len(features)==3

    feat0_shape = features[0].shape
    print('Feature shape for the 1st window: ', feat0_shape)
    assert feat0_shape[0]>0

    feat1_shape = features[1].shape
    print('Feature shape for the 2nd window: ', feat1_shape)
    assert feat1_shape[0]>0

    assert len(features[0])==len(features[1]), 'hog result is expected to be the same between all windows. len(feature[0]) = {},  len(features[1]: {}.'.format(len(features[0]) , len(features[1]) ) 
    frame_feature_size = len(features[0]) 
    assert frame_feature_size==training_feature_num, 'hog result is expected to be the same between training and video frame. frame_feature_size: {}, training_feature_num: {}'.format(frame_feature_size, training_feature_num)


    #####################################################
    # Hog() Test 
    #####################################################
    print('\n######################### Hog() Test ############################\n')
    logger.info(' HogExtractor Hog() Test ')
    car_brg = cv2.imread('data/vehicles/GTI_Right/image0177.png')
    noncar_brg = cv2.imread('data/non-vehicles/GTI/image155.png')

    car = cv2.cvtColor(car_brg, cv2.COLOR_BGR2RGB)
    noncar = cv2.cvtColor(noncar_brg, cv2.COLOR_BGR2RGB)
    visualize([[car, noncar]], [['Car image', 'Non-car Image']], 'data/outputs/car_not_car.png' )

    car = cv2.cvtColor(car_brg, cv2.COLOR_BGR2YCrCb)
    noncar = cv2.cvtColor(noncar_brg, cv2.COLOR_BGR2YCrCb)
    # car = cv2.cvtColor(car_brg, cv2.COLOR_BGR2HLS)
    # noncar = cv2.cvtColor(noncar_brg, cv2.COLOR_BGR2HLS)

    orient=9
    pix_per_cell=8
    cell_per_block=2

    img_list = []
    title_list = []
    img = car[:,:,0]
    car_ch1_feat, car_ch1_hog = get_hog_features(img,   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)

    img = noncar[:,:,0]
    noncar_ch1_feat, noncar_ch1_hog = get_hog_features(img,   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)


    img_list.append([car[:,:,0], car_ch1_hog, noncar[:,:,0], noncar_ch1_hog])
    title_list.append(['Car Ch1', 'Car Ch1 Hog', 'Non-car Ch1', 'Non-car Ch1 Hog'])



    
    img = car[:,:,1]
    car_Ch2_feat, car_Ch2_hog = get_hog_features(img,   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)

    img = noncar[:,:,1]
    noncar_Ch2_feat, noncar_Ch2_hog = get_hog_features(img,   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)


    img_list.append([car[:,:,1], car_Ch2_hog, noncar[:,:,1], noncar_Ch2_hog])
    title_list.append(['Car Ch2', 'Car Ch2 Hog', 'Non-car Ch2', 'Non-car Ch2 Hog'])


    img = car[:,:,2]
    car_Ch3_feat, car_Ch3_hog = get_hog_features(img,   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)

    img = noncar[:,:,2]
    noncar_Ch3_feat, noncar_Ch3_hog = get_hog_features(img,   orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        vis=True, feature_vec=True)


    img_list.append([car[:,:,2], car_Ch3_hog, noncar[:,:,2], noncar_Ch3_hog])
    title_list.append(['Car Ch3', 'Car Ch3 Hog', 'Non-car Ch3', 'Non-car Ch3 Hog'])

    
    visualize(img_list, title_list, 'data/outputs/HOG_example.jpg')
    



    print('\n**************** All Tests Passed! *******************')



def visualize(rbg_img_matrix, title_matrix, save_to_file='', enable_show= False):
    """show images in a grid layout
    
    Args:
        rbg_img_matrix (LIST): a list of row X column grid, 
                                e.g. [ [img1, img2, img3], 
                                       [img11, img22, img33] ]
        title_matrix (LIST): Same shape as img matrix
        save_to_file (str, optional): Description
    
    Returns:
        None: None
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    assert len(rbg_img_matrix)==len(title_matrix)
    row_len = len(rbg_img_matrix)
    col_len = len(rbg_img_matrix[0])
    assert len(rbg_img_matrix[0]) == len(title_matrix[0]), 'Expect a row has the same number of images and titles'


    f, axes = plt.subplots(row_len, col_len)

    f.tight_layout()

    for row in range(row_len):

        assert len(rbg_img_matrix[row]) == len(title_matrix[row]) , 'Expect every row has the same number of images and title'
        assert len(rbg_img_matrix[row]) == col_len , 'Expect every row has the same length'

        for col in range(col_len):

            if row_len == 1:
                ax = axes[col]
            else:
                ax = axes[row][col]
                
            img = rbg_img_matrix[row][col]
            title = title_matrix[row][col]
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')


    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if '' != save_to_file: 
        plt.savefig(save_to_file)

    if enable_show:
        plt.show()
    plt.close()   




if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))