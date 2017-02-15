from VehicleDetector import VehicleDetector
from moviepy.editor import VideoFileClip
import cv2


vehicle_detector = VehicleDetector( enable_checkpoint=True,
                                    car_path='data/vehicles/', noncar_path = 'data/non-vehicles/',
                                    feat_color_conv=cv2.COLOR_BGR2YCrCb, 
                                    orient=9, pix_per_cell=8, cell_per_block=2,  spatial_shape=(32, 32), hist_bins=32, 
                                    filter_maxcount=5, 
                                    heat_threhold=15, 
                                    win_scale_list= [1,             1.5,            2,        ], 
                                    ROI_list=       [(0.52,0.7),    (0.52,0.85),    (0.7,1),  ],
                 )

DBG_counter=0

def process_image(img):
    global DBG_counter
    


    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite('data/debug_all_images/input/'+ str(DBG_counter)+'.jpg', img_bgr)

    img_bgr = vehicle_detector.hightlightCars(img_bgr)

    cv2.imwrite('data/debug_all_images/output/'+ str(DBG_counter)+'.jpg', img_bgr)
    DBG_counter += 1


    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)




    return img_rgb


def main():


    test_output =   'data/project.mp4' # 'data/project.mp4' #

    clip = VideoFileClip('data/project_video.mp4')  #test_video.mp4   project_video.mp4

    test_clip = clip.fl_image(process_image)

    test_clip.write_videofile(test_output, audio=False)

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))