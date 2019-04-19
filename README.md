
# This is a toy project for fun!


# Demo

![](https://github.com/felixchenfy/Data-Storage/raw/master/Detect-Hand-Grasping-Object/demo-Detect-Hand-Grasping-Object.gif)

# Details

**What's this?**  
The program can **detect me grasping the object**, and then mask the grasped object with box and red mask.

**Algorithms:**  
I used Mask RCNN to detect objects in the first frame of the video. Then use SiamMask to track the objects. Meanwhile, OpenPose is used to detect my hand wrist.  
If my hand coincides with the object, it's considered to be a grasp, and the object is added with a mark.


**Warning:**  
This project is just for fun. No practical usage.

# How I made this demo video (How to run code)

1. Adjust my laptop's camera to a proper angle as shown in the demo. Put 2 objects on table.
2. Run this file "demo_maskrcnn.ipynb". It reads an image from the camera, detect the objects, and save their bounding box positions into "boxes_pos.txt".
3. Run "$ python main_detect_hand_grasping_object.py". The program starts tracking objects and my hands.   
Meanwhile, as you can see, I used my smart phone to record the video.



# Install environment

Please first clone my project and cd into it.   
Then I followed the following steps sequencially to setup the environment for the 3 github repos that I used.

(References: SiamMask, Mask_RCNN, tf-pose-estimation. Thanks!)

```

git clone https://github.com/foolwood/SiamMask
git clone https://github.com/ildoonet/tf-pose-estimation
git clone https://github.com/matterport/Mask_RCNN

export MyRoot=$PWD
cd SiamMask && export SiamMask=$PWD && export PYTHONPATH=$PWD:$PYTHONPATH && cd ..
cd $MyRoot

conda create -n tf tensorflow-gpu
conda activate tf

# 1. Install for tf-pose-estimation
pip install -r doc/requirements1.txt # for tf-pose-estimation
cd tf-pose-estimation/tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
cd $MyRoot


# 2. Install for Mask RCNN
conda install jupyter
conda install keras
cd Mask_RCNN
python setup.py install
git clone  https://github.com/waleedka/coco  
cd coco/PythonAPI  
pip install ninja yacs tqdm　# maskrcnn_benchmark and coco api dependencies  
python setup.py build_ext install  
cd $MyRoot

# 3. Install for SiamMask
pip install -r doc/requirements2.txt
cd SiamMask && bash make.sh
cd $MyRoot

```
 


# Test if install is good

## (1) SiamMask
$ python demo_siammask.py

```
Test official example:
$ cd $SiamMask/experiments/siammask    
$ python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json  
$ python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_my_bottle.json --base_path ../../data/my_bottle  
```

## (2) Mask RCNN
see: demo.ipynb

## (3) OpenPose
$ python demo_openpose.py

## (4) Yolo (Not used)
```
$ ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights ../images/00001.png  
$ ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights ../images2/00001.png  
```