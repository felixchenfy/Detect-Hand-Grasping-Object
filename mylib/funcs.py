
import numpy as np
import cv2
import math
from os import listdir
from os.path import isfile, join

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

def get_filenames(path, sort = False):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    if sort:
        onlyfiles.sort()
    return onlyfiles
    

class OneObjTracker(object):
    def __init__(self):
        self.reset()

    def track(self, skelsList):
        N = len(skelsList)
        if self.prev_skel is None:
            res_idx = 0 # default is zero
        else:
            
            dists = [0]*N
            for i, skel in enumerate(skelsList):
                dists[i] = self.measure_dist(self.prev_skel, skel)
            min_dist = min(dists)
            min_idx = dists.index(min_dist)
            res_idx = min_idx
        self.prev_skel = skelsList[res_idx].copy()
        return res_idx

    def measure_dist(self, prev_skel, curr_skel):
        neck = 1*2
        dx2 = (prev_skel[neck] - curr_skel[neck])**2
        dy2 = (prev_skel[neck+1] - curr_skel[neck+1])**2
        return math.sqrt(dx2+dy2)

    def reset(self):
        self.prev_skel = None