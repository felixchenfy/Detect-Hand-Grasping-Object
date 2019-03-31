# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
import sys, os, cv2
import numpy as np
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
# sys.path.append(CURR_PATH)  # To find local version of the library\n",
import torch, argparse
import SiamMask.tools.test as libsiam
from mylib.siammask import SiamMaskTracker, select_initial_object_pos
from mylib.io import DataLoader_folder

if __name__ == '__main__':
   
    # -- Create model
    model_siammask = SiamMaskTracker()

    # -- Set input images
    images_loader = DataLoader_folder(CURR_PATH + "images/")

    # -- Loop
    toc = 0
    ith_img = 0
    while ith_img <= images_loader.num_images:
        ith_img += 1
        tic = cv2.getTickCount()
        img, _, _ = images_loader.load_next_image()
        image_disp = img.copy()
        
        if ith_img == 1:  # init
            x, y, w, h = select_initial_object_pos(img)
            model_siammask.init_tracker(img, x, y, w, h)

        elif ith_img > 1:  # tracking
            mask, location = model_siammask.track(img)
            model_siammask.draw_mask_onto_image(image_disp, mask, location)
        

        # -- Draw results
        if 1: # Draw resultz
            cv2.imshow('SiamMask', image_disp)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
