
# ======================================================================
# 1. Install environment

Please first clone my project and cd into it.   
Then I followed the following steps sequencially to setup the environment for the 3 github repos that I used.

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
 


# ======================================================================
# 2. Test

## 2.1 .SiamMask
$ python demo_siammask.py

```
Test official example:
$ cd $SiamMask/experiments/siammask    
$ python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json  
$ python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_my_bottle.json --base_path ../../data/my_bottle  
```

## 2.2. Mask RCNN
see: demo.ipynb

## 2.3. OpenPose
$ python demo_openpose.py

## 2.4. Yolo (Not used)
```
$ ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights ../images/00001.png  
$ ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights ../images2/00001.png  
```


# ======================================================================
# Others

## Versions
* SiamMask work with cpu:  
https://github.com/foolwood/SiamMask/issues/15


## Installation problems

* Torch not working with python 3.7:
https://github.com/HazyResearch/metal/issues/101
Use a specific version: torch==0.4.1.post2