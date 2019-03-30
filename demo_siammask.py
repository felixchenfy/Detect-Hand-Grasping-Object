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


def set_SiamMask_command_line_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
    model_folder = 'SiamMask/experiments/siammask/'
    parser.add_argument('--resume', type=str, required=False,
                        default = model_folder + 'SiamMask_DAVIS.pth',
                        metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config', default= model_folder + 'config_davis.json',
                        help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default='SiamMask/data/my_bottle', help='datasets')
    args = parser.parse_args()
    return args 

class init_SiamMask_model(object):

    def __init__(self):
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        args = set_SiamMask_command_line_arguments()
        cfg = libsiam.load_config(args)
        from custom import Custom
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

    def init_tracker(self, im, x, y, w, h):
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = libsiam.siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp'])  # init tracker
    
    def track(self, im):
        self.state = libsiam.siamese_track(self.state, im, mask_enable=True, refine_enable=True)  # track
        location = self.state['ploygon'].flatten()
        ''' my notes:
        print("\nstate:", state)
        print("\nlocation:", location) # [364.17523 417.03296 240.6001  348.26935 365.40112 123.98926 488.97626 192.75287]
        '''
        mask = self.state['mask'] > self.state['p'].seg_thr
        return mask, location

if __name__ == '__main__':
   
    model_siammask = init_SiamMask_model()

    # Parse Image file
    # img_files = sorted(glob.glob(os.path.join(args.base_path, '*.jp*')))
    image_folder = CURR_PATH + "images_bottle/"
    img_files = sorted(glob.glob( image_folder + '*.png'))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()

        if f == 0:  # init
            model_siammask.init_tracker(im, x, y, w, h)

        elif f > 0:  # tracking
            mask, location = model_siammask.track(im)
          
            if 1: # Draw resultz
                im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                cv2.imshow('SiamMask', im)
                key = cv2.waitKey(1)
                if key > 0:
                    break

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
