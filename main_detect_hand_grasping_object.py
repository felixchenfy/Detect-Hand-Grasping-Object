
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

# Functions =============================================================

def get_objects_bboxes():
    import simplejson
    filename = 'boxes_pos.txt'
    with open(filename, 'r') as f:
        bboxes = simplejson.load(f)
        return bboxes
    exit()
    

def unload_bbox_pos(b):
    x, y, w, h = b[0],b[1],b[2],b[3]
    return x, y, w, h

def update_bbox(bboxes, grasped_obj, new_loc):
    # new_loc = [364.17523,417.03296,240.6001,348.26935,365.40112,123.98926,488.97626,192.75287]
    xs = new_loc[::2]
    ys = new_loc[1::2]
    bboxes[grasped_obj] = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]

def get_bbox_center(bbox):
    return (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))


def draw_bbox_onto_image(image_disp, b):
    minx, miny, maxx, maxy = b[0], b[1], b[0]+b[2], b[1]+b[3]
    image_disp = cv2.rectangle(image_disp,(minx, miny),(maxx, maxy),(0,255,0), 4)

if __name__ == '__main__':
   
    # Set input =================================

    # -- Set input images
    FROM_WEBCAM = True
    if FROM_WEBCAM:
        images_loader = DataLoader_usbcam(max_framerate = 8)
    elif FROM_FOLDER:
        images_loader = DataLoader_folder(CURR_PATH + "images/", 0)

    # -- Openpose
    openpose_image_size = "304x240" # 14 fps
    # openpose_image_size = "240x208" 
    OpenPose_MODEL = ["mobilenet_thin", "cmu"][1]
    model_openpose = SkeletonDetector(OpenPose_MODEL, openpose_image_size)

    # -- SiamMask
    models_siammask = []
    bboxes = get_objects_bboxes()
    N_obj = len(bboxes)
    for i in range(N_obj): # Init a tracker for every detected object
        a_new_tracker = SiamMaskTracker()
        models_siammask.append(a_new_tracker)

    # -- Loop
    ith_img = 0
    WINDOW_NAME = "CV2_WINDOW"
    while ith_img <= images_loader.num_images:
        ith_img += 1

        # -- Load image
        img, _, _ = images_loader.load_next_image()
        image_disp = img.copy()
        

        # -- Detect human skeleton and hand
        humans = model_openpose.detect(img)
        hands = model_openpose.get_hands_in_xy(humans)

        # -- Track object
        if ith_img == 1:  # input object location for tracking
            for i in range(N_obj):
                # x, y, w, h = select_initial_object_pos(img, object_index=i)
                x, y, w, h = unload_bbox_pos(bboxes[i])
                models_siammask[i].init_tracker(img, x, y, w, h)

        elif ith_img > 1:  # tracking if an object is grasped

            for i in range(N_obj):
                mask, location = models_siammask[i].track(img)
                update_bbox(bboxes, i, location)  # update bboxes location
                


        # -- Detect grasp state
        def is_near(hand, box):
            hx, hy = hand[0], hand[1]
            x0, y0, x1, y1 = box[0], box[1], box[0]+box[2], box[1]+box[3]
            is_in_range = lambda x, l, h: x>=l and x<=h
            if is_in_range(hx, x0, x1) and is_in_range(hy, y0, y1):
                return True
            else:
                return False

        def is_near2(hand, mask):
            r, c = int(hand[1]), int(hand[0])
            return mask[r, c]!=0

        for hand in hands:
            for i, box in enumerate(bboxes):
                if is_near(hand, box):
                    models_siammask[i].draw_mask_onto_image(image_disp,
                        models_siammask[i].mask, models_siammask[i].location) # draw rotated mask
                    # draw_bbox_onto_image(image_disp, bboxes[i])
                    print("hand:",hand,", bbox:",box,)
                    print("... hand is near!")
                else:
                    print("hand:",hand,", bbox:",box,)

   

        # -- Draw
        if 0:
            if 1: # Draw skeleton
                model_openpose.draw(image_disp, humans) # draw skeleton
                for hand in hands: # draw hands
                    cv2.circle(image_disp, center=(hand[0], hand[1]), radius = 10,
                        color = [0, 0, 255], thickness=2, lineType=cv2.LINE_AA) 
                model_openpose.draw_fps(image_disp)
    
            if 0: # Draw object position
                for b in bboxes:
                    cv2.circle(image_disp, center=get_bbox_center(b), radius = 10,
                        color = [255, 0, 0], thickness=2, lineType=cv2.LINE_AA) 

            if 1: # Draw object bounding box
                for i, box in enumerate(bboxes):
                    draw_bbox_onto_image(image_disp, bboxes[i])

        if 1: # Display
            image_disp = cv2.resize(image_disp,(0,0),fx=1.5,fy=1.5) # resize to make picture bigger
            cv2.imshow(WINDOW_NAME, image_disp)
            cv2.imwrite("result_images/{:05d}.jpg".format(ith_img), image_disp)
            q = cv2.waitKey(1)
            if q!=-1 and chr(q) == 'q':
                break

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
