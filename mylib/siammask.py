

import glob
import sys, os, cv2
import numpy as np
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(CURR_PATH+"../")  # To find local version of the library\n",
import torch, argparse
import SiamMask.tools.test as libsiam
sys.path.append(CURR_PATH+"../SiamMask/experiments/siammask")  # To find local version of the library\n",
from custom import Custom

# -- Set model path and configuration file's path
class FakeArgParser(object):
    def __init__(self):
        model_folder = 'SiamMask/experiments/siammask/'
        self.resume = model_folder + 'SiamMask_DAVIS.pth'
        self.config = model_folder + 'config_davis.json'
        self.arch = None

# -- Init model
class SiamMaskTracker(object):

    def __init__(self):
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        args = FakeArgParser()
        cfg = libsiam.load_config(args)
        siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            assert libsiam.isfile(args.resume), '{} is not a valid file'.format(args.resume)
            siammask = libsiam.load_pretrain(siammask, args.resume)

        siammask.eval().to(device)
        
        # -- Output
        self.siammask = siammask
        self.args = args
        self.cfg = cfg
        self.state = None

    def init_tracker(self, img, x, y, w, h):
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = libsiam.siamese_init(img, target_pos, target_sz, self.siammask, self.cfg['hp'])  # init tracker
    
    def track(self, img):
        self.state = libsiam.siamese_track(self.state, img, mask_enable=True, refine_enable=True)  # track
        location = self.state['ploygon'].flatten()
        ''' my notes:
        print("\nstate:", state)
        print("\nlocation:", location) # [364.17523 417.03296 240.6001  348.26935 365.40112 123.98926 488.97626 192.75287]
        '''
        mask = self.state['mask'] > self.state['p'].seg_thr
        self.mask = mask
        self.location = location
        return mask, location

    def draw_mask_onto_image(self, image_disp, mask, location):
        image_disp[:, :, 2] = (mask > 0) * 255 + (mask == 0) * image_disp[:, :, 2]
        cv2.polylines(image_disp, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

def select_initial_object_pos(img, object_index = 0):
    if 0:
        cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            init_rect = cv2.selectROI('SiamMask', img, False, False)
            x, y, w, h = init_rect
        except:
            exit()
    else:
        import simplejson
        filename = 'boxes_pos.txt'

        with open(filename, 'r') as f:
            bboxes = simplejson.load(f)

        b = bboxes[object_index]
        x, y, w, h = b[0],b[1],b[2],b[3]

    return x, y, w, h 