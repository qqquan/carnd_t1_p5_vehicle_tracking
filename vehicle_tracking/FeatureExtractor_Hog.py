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
            blk_pos_br_row = br_row//self.pix_per_cell
            blk_pos_br_col = br_col//self.pix_per_cell


            feat1 = hog1[blk_pos_ul_row:blk_pos_br_row, blk_pos_ul_col:blk_pos_br_col ].reshape(-1)
            feat2 = hog2[blk_pos_ul_row:blk_pos_br_row, blk_pos_ul_col:blk_pos_br_col ].reshape(-1)
            feat3 = hog3[blk_pos_ul_row:blk_pos_br_row, blk_pos_ul_col:blk_pos_br_col ].reshape(-1)

            feat = np.concatenate((feat1, feat2, feat3))

            features.append(feat)


            logger.debug('HogExtractor - Window Position - x - {}:{}.   y - {}:{} '.format(ul_col, br_col, ul_row, br_row))
            logger.debug('HogExtractor - HOG Block Position - x - {}:{}.   y - {}:{} '.format(blk_pos_ul_col, blk_pos_br_col, blk_pos_ul_row, blk_pos_br_row))

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
    logger.info('######################### Training Image Test')
    training_img_brg = cv2.imread('data/vehicles/GTI_Right/image0025.png')

    img = training_img_brg
    print('Training Image shape: ', img.shape)

    w1 = ((0,0),img.shape[:2])
    windows = [w1]

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
    assert feat0_size==(feature_num), 'Expected feature number: {}'.format(feature_num)




    #####################################################
    # Video Frame
    #####################################################
    print('\n\n######################### Video Frame Test ############################ \n')
    logger.info('######################### Video Frame Test')
    video_img_brg = cv2.imread('data/test_images/test6.jpg')
    img = video_img_brg
    print('Video frame shape: ', img.shape)




    window_size = 64
    w1 = ((0,0),(window_size,window_size))
    w2 = ((400,600),(400+window_size, 600+window_size))
    w3 = ((504,0),(504+window_size, 0+window_size*2))
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

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))