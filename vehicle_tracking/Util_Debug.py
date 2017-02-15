import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


    assert len(rbg_img_matrix)==len(title_matrix)
    row_len = len(rbg_img_matrix)
    col_len = len(rbg_img_matrix[0])
    assert len(rbg_img_matrix[0]) == len(title_matrix[0]), 'Expect a row has the same number of images and titles'

    assert len(rbg_img_matrix[0]) != 0
    assert len(title_matrix[0]) != 0
    
    f, axes = plt.subplots(row_len, col_len)

    f.tight_layout()

    for row in range(row_len):

        assert len(rbg_img_matrix[row]) == len(title_matrix[row]) , 'Expect every row has the same number of images and title'
        assert len(rbg_img_matrix[row]) == col_len , 'Expect every row has the same length'

        for col in range(col_len):

            if row_len == 1:
                if col_len == 1:
                    ax = axes
                else: 
                    ax = axes[col]
            else:
                if col_len is 1:
                    ax = axes[row]
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
        plt.show( )
    plt.close()   