
import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math

# Mine ==============================================================

from mylib.displays import drawActionResult
from mylib.io import DataLoader_usbcam, DataLoader_folder
import mylib.funcs as myfunc
# import mylib.feature_proc as myproc 
# from mylib.action_classifier import *

# Openpose ==============================================================
from mylib.openpose import SkeletonDetector
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# Set input =================================

FROM_WEBCAM = True

if FROM_WEBCAM:
    save_idx = "4"
    DO_INFER_ACTIONS =  False
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    openpose_image_size = "304x240" # 14 fps
    # openpose_image_size = "240x208" # 14 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]

elif FROM_FOLDER:
    save_idx = "5"
    DO_INFER_ACTIONS =  False
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    def set_source_images_from_folder():
        # return CURR_PATH + "../data_test/apple/", 1
        # return CURR_PATH + "../data_test/mytest/", 0
        return "/home/qiancheng/DISK/feiyu/TrainYolo/data_yolo/video_bottle/images/", 2
    SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES = set_source_images_from_folder()
    save_idx += SRC_IMAGE_FOLDER.split('/')[-2] # plus file name
    openpose_image_size = "304x240" # 14 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]

else:
    assert False

# ==============================================================



if __name__ == "__main__":
 
    # -- Detect sekelton
    my_detector = SkeletonDetector(OpenPose_MODEL, openpose_image_size)

    # -- Load images
    if FROM_WEBCAM:
        images_loader = DataLoader_usbcam()
    elif FROM_FOLDER:
        images_loader = DataLoader_folder(SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES)

    # -- Loop through all images
    ith_img = 1
    while ith_img <= images_loader.num_images:
        img, action_type, img_info = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n\n========================================")
        print("\nProcessing {}/{}th image\n".format(ith_img, images_loader.num_images))

        # Detect skeleton
        humans = my_detector.detect(img)
        skelsList = my_detector.humans_to_skelsList(humans)

        if len(skelsList) > 0:

            # Loop through all skeletons
            for ith_skel in range(0, len(skelsList)):
                skeleton = SkeletonDetector.get_ith_skeleton(skelsList, ith_skel)
                
                # Draw skeleton
                my_detector.draw(image_disp, humans)
                    
        else:
            classifier.reset() # clear the deque

        # Write result to png
        if 1:
            cv2.imwrite(CURR_PATH+"result_images/" 
                + myfunc.int2str(ith_img, 5)+".png", image_disp)

        if 1: # Display
            image_disp = cv2.resize(image_disp,(0,0),fx=1.5,fy=1.5) # resize to make picture bigger
            cv2.imshow("action_recognition", image_disp)
            q = cv2.waitKey(1)
            if q!=-1 and chr(q) == 'q':
                break

        # Loop
        print("\n")
        ith_img += 1

