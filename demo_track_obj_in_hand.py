
import glob
import sys, os
import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# Openpose ==============================================================

from mylib.displays import drawActionResult
from mylib.io import DataLoader_usbcam, DataLoader_folder
import mylib.funcs as myfunc
from mylib.openpose import SkeletonDetector

# SiamMask ==============================================================
import torch, argparse
import SiamMask.tools.test as libsiam
from mylib.siammask import SiamMaskTracker, select_initial_object_pos

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
        return CURR_PATH + "images/", 0
    SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES = set_source_images_from_folder()
    save_idx += SRC_IMAGE_FOLDER.split('/')[-2] # plus file name
    openpose_image_size = "304x240" # 14 fps
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]

else:
    assert False

if __name__ == '__main__':
   
    # -- Create model
    model_siammask = SiamMaskTracker()
    model_siammask2 = SiamMaskTracker()
    model_openpose = SkeletonDetector(OpenPose_MODEL, openpose_image_size)

    # -- Set input images
    images_loader = DataLoader_folder(CURR_PATH + "images/")

    # -- Loop
    ith_img = 0
    WINDOW_NAME = "CV2_WINDOW"
    while ith_img <= images_loader.num_images:
        ith_img += 1

        # -- Load image
        img, _, _ = images_loader.load_next_image()
        image_disp = img.copy()
        
        # -- Track object
        if ith_img == 1:  # init the object loc to track
            
            x, y, w, h = select_initial_object_pos(img, object_index=0)
            model_siammask.init_tracker(img, x, y, w, h)

            x, y, w, h = select_initial_object_pos(img, object_index=1)
            model_siammask2.init_tracker(img, x, y, w, h)

        elif ith_img > 1:  # tracking

            mask, location = model_siammask.track(img)
            model_siammask.draw_mask_onto_image(image_disp, mask, location) # draw tracked object

            mask, location = model_siammask2.track(img)
            model_siammask2.draw_mask_onto_image(image_disp, mask, location) # draw tracked object

        # -- Detect human skeleton and hand
        humans = model_openpose.detect(img)
        hands = model_openpose.get_hands_in_xy(humans)

        # -- Draw skeleton
        if 1:
            model_openpose.draw(image_disp, humans) # draw skeleton
            for hand in hands: # draw hands
                cv2.circle(image_disp, center=(hand[0], hand[1]), radius = 10,
                    color = [255, 0, 0], thickness=2, lineType=cv2.LINE_AA) 
            model_openpose.draw_fps(image_disp)

        if 1: # Display
            image_disp = cv2.resize(image_disp,(0,0),fx=1.5,fy=1.5) # resize to make picture bigger
            cv2.imshow(WINDOW_NAME, image_disp)
            q = cv2.waitKey(1)
            if q!=-1 and chr(q) == 'q':
                break

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
