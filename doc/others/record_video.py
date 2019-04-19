#!/usr/bin/env python
# -*- coding: utf-8 -*-


# -- Standard
import numpy as np
import sys, os
import cv2
import datetime
from my_lib import *

PYTHON_FILE_PATH=os.path.join(os.path.dirname(__file__))+"/"

# ----------------------------------------------------


if __name__=="__main__":

    save_dir = "usbcam_images/"
    framerate = 10

    # ---------------
    processor = ProcessEvent(folder_name = "none")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)

    i=0
    while(True):
        # Capture frame-by-frame
        ret, image = cap.read()

        # Show image
        cv2.imshow("human_pose_recorder", image)
        
        # Process key event
        key = cv2.waitKey(10) 
        processor.process_event(key, image)
        if(key>=0 and chr(key)=='q'):
            break
            

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

