
import numpy as np
import cv2
import glob
import sys, os
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

FROM_VIDEO = False
if FROM_VIDEO:
    cap = cv2.VideoCapture('vtest.avi')
else:
    image_folder = CURR_PATH + "images_meter/"
    get_file_name = lambda idx: "image" + int2str(idx, 5)+".png"
    image_folder_out = image_folder[:-1] + "_res/"

fgbg = cv2.createBackgroundSubtractorMOG2()

for idx_image in range(1, 400+1):
    if FROM_VIDEO:
        ret, frame = cap.read()
    else:
        filename = image_folder + get_file_name(idx_image)
        print("Reading {}".format(filename),)
        frame = cv2.imread(filename)
    print("{}th image".format(idx_image))
    
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(100)
    if k!=-1 and chr(k)=='q':
        break
cap.release()
cv2.destroyAllWindows()
