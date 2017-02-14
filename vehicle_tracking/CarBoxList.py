import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.info('CarBoxList module loaded')


def doOverlap(np_box1, np_box2):
    #TODO: either box is on left or above to the other, then false; write test code 

    is_overlapping_detected = False

    b1_ul_pos = np_box1[0]
    b1_br_pos = np_box1[1]

    b1_ul_y = b1_ul_pos[0]
    b1_ul_x = b1_ul_pos[1]

    b1_br_y = b1_br_pos[0]
    b1_br_x = b1_br_pos[1]



    b2_ul_pos = np_box2[0]
    b2_br_pos = np_box2[1]

    b2_ul_y = b2_ul_pos[0]
    b2_ul_x = b2_ul_pos[1]

    b2_br_y = b2_br_pos[0]
    b2_br_x = b2_br_pos[1]

    if b1_br_x < b2_ul_x:
        #box1 is on the left of box2
        # logger.debug('doOverlap : box1 is on the left of box2. b1_br_x={}, b2_ul_x={}'.format(b1_br_x, b2_ul_x) )
        is_overlapping_detected = False
    elif b1_ul_x>b2_br_x:
        #box1 is on right
        # logger.debug('doOverlap: box1 is on right')
        is_overlapping_detected = False
    elif b1_br_y < b2_ul_y:
        #box1 is over box2
        # logger.debug('doOverlap : box1 is over box2')
        is_overlapping_detected = False
    elif b1_ul_y > b2_br_y:
        # logger.debug('doOverlap : box1 is under box2')
        #box1 is under box2
        is_overlapping_detected = False
    else:
        # logger.debug('overlapped ')
        is_overlapping_detected = True


    return is_overlapping_detected


# invalid if the new box does not overlap any of the past boxes
def isNewBoxValid(newbox, past_box_list):
    is_valid = True
    for past_box in past_box_list:
        if doOverlap(newbox, past_box) == False:
            is_valid = False
            break
    return is_valid

class CarBoxList():
  
    def __init__(self, max_count=5):

        self.MAXCOUNT=max_count
        self.car_list = []

    def update(self, new_box_list):
        """Summary
        
        Args:
            new_box_list (TYPE): a list of Numpy Arrays
        
        Returns:
            TYPE: Description
        """
        if len(new_box_list) != len(self.car_list):

            self.reset(new_box_list)

        elif len(new_box_list) > 0:

            for new_box, car_pastboxes in zip(new_box_list, self.car_list):
                #find if the new box is valid for the car
                is_new_box_valid = True
                for past_box in car_pastboxes:
                    if doOverlap(new_box, past_box) == False:
                        self.reset(new_box_list)
                        is_new_box_valid = False

                if is_new_box_valid:
                    car_pastboxes.append(new_box)
                    if len(car_pastboxes)>self.MAXCOUNT:
                        car_pastboxes.pop(0)
                else:
                    self.reset(new_box_list)
                    #no need to look at the rest of new box list
                    break
 
             
        else:
            # if no car is found, and car_list is empty, do nothing
            pass


    def getBoxList(self):
        box_list = []
        for car_boxes in self.car_list:
            num = len(car_boxes)
            if num==0:
                logger.debug( 'CarBoxList - self.car_list:  {}'.format(self.car_list))
                continue

            total = np.zeros_like(car_boxes[0])
            for box in  car_boxes:
                total = np.add(total, box)

            avg_box = total/num  

            box_list.append(avg_box)

        return box_list

    def reset(self, box_list=[]):

        self.car_list = []

        if len(box_list)>0:
            for box in box_list:
                self.car_list.append([box])


    def getNumOfCars(self):
        return len(self.car_list)





def main():
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('######################### CarboxList - Module Test ############################')

    print('\n-- Test0: doOverlap() --------------------------------- \n')


    box1 = np.array( ((0, 1), (9,10)))
    box2 = np.array( ((0, 10), (9,19)))

    assert doOverlap(box1, box2) == True 
    assert doOverlap(box1, box2) == True 

    assert doOverlap(box1, box1) == True

    # left - right
    box1 = np.array( ((0, 1), (10,11)))
    box2 = np.array( ((0, 100), (9,111)))
    assert doOverlap(box1, box2) == False
    assert doOverlap(box2, box1) == False

    # under and above
    box1 = np.array( ((0, 100), (10,111)))
    box2 = np.array( ((11, 100), (22,111)))
    assert doOverlap(box1, box2) == False
    assert doOverlap(box2, box1) == False
    print('\n-- Test1: doOverlap() --------------------------------- \n')



    carbox_list = CarBoxList()
    box1 = ((0, 0), (10,10))
    box2 = ((1, 1), (11,11))
    np_box1= np.array(box1)
    np_box2= np.array(box2)
    np_box_list = [np_box1, np_box2] 

    carbox_list.update(np_box_list)


    print('num of cars: ', carbox_list.getNumOfCars())
    car_list = carbox_list.getBoxList() 
    print('car_list:', car_list)
    print('first car box :', car_list[0])
    assert carbox_list.getNumOfCars() == len(np_box_list), 'num of cars: {}, num of inputs: {}'.format(carbox_list.getNumOfCars() , len(np_box_list))

    car_num_of_past_boxes = len(carbox_list.car_list[0])
    print(' num of past boxes for the first car:', car_num_of_past_boxes)
    assert car_num_of_past_boxes == 1, 'car_num_of_past_boxes={}'.format(car_num_of_past_boxes)


    print('\n-- Test2: add the same boxes again, the number of boxes for a car should be doubled. \n')
    np_box1= np.array(box1)
    np_box2= np.array(box2)
    np_box_list = [np_box1, np_box2] 

    carbox_list.update(np_box_list)


    print('num of cars: ', carbox_list.getNumOfCars())
    car_list = carbox_list.getBoxList() 
    print('car_list:', car_list)
    print('first car box :', car_list[0])

    assert carbox_list.getNumOfCars() == len(np_box_list), 'num of cars: {}, num of inputs: {}'.format(carbox_list.getNumOfCars() , len(np_box_list))
    car_num_of_past_boxes = len(carbox_list.car_list[0])
    print(' num of past boxes for the first car:', car_num_of_past_boxes)
    assert car_num_of_past_boxes == 2, 'car_num_of_past_boxes={}'.format(car_num_of_past_boxes)




    print('\n-- Test3: add the fewer or more cars , the number of boxes for a car should reset to 1 \n')
    box1 = ((1, 2), (11,10))

    np_box1= np.array(box1)
    np_box_list = [np_box1] 


    carbox_list.update(np_box_list)

    print('num of cars: ', carbox_list.getNumOfCars())
    car_list = carbox_list.getBoxList() 
    print('car_list:', car_list)
    print('first car box :', car_list[0])

    assert carbox_list.getNumOfCars() == len(np_box_list), 'num of cars: {}, num of inputs: {}'.format(carbox_list.getNumOfCars() , len(np_box_list))
    car_num_of_past_boxes = len(carbox_list.car_list[0])
    print(' num of past boxes for the first car:', car_num_of_past_boxes)
    assert car_num_of_past_boxes == 1, 'car_num_of_past_boxes={}'.format(car_num_of_past_boxes)




    print('\n-- Test4: filtering --------------------------------\n')
    logger.info('\n-- Test4: filtering --------------------------------\n')
    box1 = ((2, 6), (6,10))

    np_box1= np.array(box1)
    np_box_list = [np_box1] 

    carbox_list.update(np_box_list)
    box1 = ((3, 7), (5,9))
    
    np_box1= np.array(box1)
    np_box_list = [np_box1] 

    carbox_list.update(np_box_list)

    print('num of cars: ', carbox_list.getNumOfCars())
    car_list = carbox_list.getBoxList() 
    print('car_list:', car_list)
    print('first car box :', car_list[0])
    assert np.average(car_list[0]) == 6

    car_num_of_past_boxes = len(carbox_list.car_list[0])
    print(' num of past boxes for the first car:', car_num_of_past_boxes)
    assert car_num_of_past_boxes == 3, 'car_num_of_past_boxes={}'.format(car_num_of_past_boxes)



    print('\n-- Test5: No car found --------------------------------- \n')
    carbox_list.update([])

    print('num of cars: ', carbox_list.getNumOfCars())
    car_list = carbox_list.getBoxList() 
    print('car_list:', car_list)
    assert carbox_list.getNumOfCars() == 0, 'num of cars: {} '.format(carbox_list.getNumOfCars() )




    print('\n**************** All Tests Passed! *******************')


if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))