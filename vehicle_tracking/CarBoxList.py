import numpy as np


def doOverlap(np_box1, np_box2):
    #TODO: either box is on left or above to the other, then false; write test code 
    return True

class CarBoxList():
  
    def __init__(self, max_count=5 ):
        self.MAXCOUNT=max_count
        self.car_list = [[]]

    def update(self, idx, box):

        np_box = np.array(box)

        if idx < len(self.car_list):
            if self.isValid(idx, np_box):
                self.filter(idx, np_box)
            else: 
                self.reset()
        else:
            self.car_list.append([np_box])

    def filter(self, idx, np_box):

        if len(self.car_list[idx]) < self.MAXCOUNT:
            self.car_list[idx].append(np_box)
        else:
            self.car_list[idx].pop(0) 
            self.car_list[idx].append(np_box)


    def isValid(self, idx, np_box):
        state = True
        if idx < len(self.car_list):
            for box in self.car_list[idx]:
                if doOverlap(box, np_box ) != True:
                    state |= False

        #TODO: check overlap etc

        return state 



    def getBoxList(self):
        box_list = []
        for car_boxes in self.car_list:
            num = len(car_boxes)

            total = np.zeros_like(car_boxes[0])
            for box in  car_boxes:
                total = np.add(total, box)

            avg_box = total/num  

            box_list.append(avg_box)

        return box_list

    def reset(self):
        print('reset')

        self.car_list = []

    def getNumOfCars(self):
        return len(self.car_list)





def main():
    carbox_list = CarBoxList()
    box1 = ((0, 0), (10,10))
    box2 = ((5, 5), (15,15))

    carbox_list.update(0, box1)
    carbox_list.update(1, box2)

    carbox_list.update(1, box1)
    carbox_list.update(0, box2)  

    print('num of cars: ', carbox_list.getNumOfCars())
    box_list = carbox_list.getBoxList() 
    print('box_list:', box_list)
    print('box :', box_list[0])
    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))