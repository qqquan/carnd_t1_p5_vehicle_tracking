import numpy as np

class FilteredHeatMap():
    def __init__(self, max_count=5, threshold=2):

        self.HEATMAP_MAXCOUNT = max_count
        self.THRESHOLD = threshold
        self.heatmap_list=[]

    def update(self, new_heatmap):
        if len(self.heatmap_list) < self.HEATMAP_MAXCOUNT:
            self.heatmap_list.append(new_heatmap)
        else:
            self.heatmap_list.pop(0) 
            self.heatmap_list.append(new_heatmap)

    def getFilteredHeatmap(self):
        num_of_maps = len(self.heatmap_list)

        total_heat = np.zeros_like(self.heatmap_list[0])
        for heat in self.heatmap_list:
            total_heat = np.add(total_heat,heat)

        avg_heat = total_heat/num_of_maps


        thresholded_heat = avg_heat
        thresholded_heat[thresholded_heat<=self.THRESHOLD] = 0

        return thresholded_heat

    def reset(self):
        self.heatmap_list=[]

def main():
    ###########################
    #### Heatmap test
    ###########################

    heatmap = FilteredHeatMap()

    heat=[]
    for i in range(3):
        heat.append( np.array([[i,i, i], [i,i, i]]) )
        heatmap.update(heat[i])

    print('filtered heatmap: ', heatmap.getFilteredHeatmap())
    assert np.sum(heatmap.getFilteredHeatmap()) == 0

    heat=[]
    for i in range(11):
        heat.append( np.array([[i,i, i], [i,i, i]]) )
        heatmap.update(heat[i])

    print('filtered heatmap: ', heatmap.getFilteredHeatmap())



    print('\n**************** All Tests Passed! *******************')

if __name__ == "__main__": 
    import time
    from datetime import timedelta

    time_start = time.time()

    main()

    time_end = time.time()
    print("Time usage: " + str(timedelta(seconds=int( time_end - time_start))))