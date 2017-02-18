from VehicleDetector import VehicleDetector
from moviepy.editor import VideoFileClip
import cv2



vehicle_detector = VehicleDetector( enable_checkpoint=False)

DBG_counter=0

def process_image(img):
    global DBG_counter
    


    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # cv2.imwrite('data/debug_all_images/input/'+ str(DBG_counter)+'.jpg', img_bgr)

    img_bgr = vehicle_detector.hightlightCars(img_bgr)

    # cv2.imwrite('data/debug_all_images/output/'+ str(DBG_counter)+'.jpg', img_bgr)
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