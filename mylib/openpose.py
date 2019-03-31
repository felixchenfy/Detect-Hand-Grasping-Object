

import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"

# Openpose ==============================================================

sys.path.append(CURR_PATH + "../tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
import tensorflow as tf

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Settings ===============================================================
DRAW_FPS = True
MAX_FRACTION_OF_GPU = 0.3

# Human pose detection ==============================================================

class SkeletonDetector(object):
    # This func is mostly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, model=None, image_size=None):
        
        if model is None:
            model = "cmu"

        if image_size is None:
            image_size = "432x368" # 7 fps
            # image_size = "336x288"
            # image_size = "304x240" # 14 fps

        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        # parser = argparse.ArgumentParser(description='tf-pose-estimation run')
        # parser.add_argument('--image', type=str, default='./images/p1.jpg')
        # parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

        # parser.add_argument('--resize', type=str, default='0x0',
        #                     help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
        # parser.add_argument('--resize-out-ratio', type=float, default=4.0,
        #                     help='if provided, resize heatmaps before they are post-processed. default=1.0')
        self.resize_out_ratio = 4.0

        # args = parser.parse_args()

        w, h = model_wh(image_size)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU
        # tf_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)#https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory

        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368),tf_config=tf_config)
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h), tf_config=tf_config)

        # self.args = args
        self.w, self.h = w, h
        self.e = e
        self.fps_time = time.time()
        self.cnt_image = 0

    def detect(self, image):
        self.cnt_image += 1
        if self.cnt_image == 1:
            self.image_h = image.shape[0]
            self.image_w = image.shape[1]
            self.scale_y = 1.0 * self.image_h / self.image_w
        t = time.time()

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)

        # Print result and time cost
        elapsed = time.time() - t
        logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        
    def draw_fps(self, img_disp):
        # logger.debug('show+')
        if DRAW_FPS:
            cv2.putText(img_disp,
                        # "Processing speed: {:.1f} fps".format( (1.0 / (time.time() - self.fps_time) )),
                        "fps = {:.1f}".format( (1.0 / (time.time() - self.fps_time) )),
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        self.fps_time = time.time()

    def humans_to_skelsList(self, humans, scale_y = 1.0): # get (x, y * scale_y)
        if scale_y is None:
            scale_y = self.scale_y
        skelsList = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_y
            skelsList.append(skeleton)
        return skelsList
    
    def get_hands(self, humans):

        
        skelsList = self.humans_to_skelsList(humans)
        def predict_hand_position(skel, idx_wrist,
                ratio = 0.4 # (wrist to hand)/(wrist to elbow)        
            ):
            idx_elbow = idx_wrist - 1
            wx, wy = skel[idx_wrist*2], skel[idx_wrist*2+1]
            ex, ey = skel[idx_elbow*2], skel[idx_elbow*2+1]
            hx = wx + (wx - ex) * ratio
            hy = wy + (wy - ey) * ratio
            return [hx, hy]

        NaN = 0
        LEFT_WRIST = 4
        RIGH_WRIST = 7
        hands = []
        for skeleton in skelsList:
            if skeleton[LEFT_WRIST] != NaN:
                hands.append(predict_hand_position(skeleton, LEFT_WRIST))
            if skeleton[RIGH_WRIST] != NaN:
                hands.append(predict_hand_position(skeleton, RIGH_WRIST))
            
        return hands 

    def get_hands_in_xy(self, humans):
        
        hands = self.get_hands(humans)

        # Change coordinate to pixel
        for i, hand in enumerate(hands):
            x = int(hand[0]*self.image_w)
            y = int(hand[1]*self.image_h)
            hands[i] = [x, y]
        return hands

    @staticmethod
    def get_ith_skeleton(skelsList, ith_skeleton=0):
        res = np.array(skelsList[ith_skeleton])
        return res