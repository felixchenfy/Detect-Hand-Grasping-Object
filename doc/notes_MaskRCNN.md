
I used this version:
https://github.com/matterport/Mask_RCNN


### Install 

* Please its own tutorial  

* Install coco api  
open my anaconda env, then:

```
# see: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md
git clone  https://github.com/waleedka/coco
cd coco/PythonAPI
# conda install ipython　# this installs the right pip and dependencies for the fresh python
pip install ninja yacs cython matplotlib tqdm　# maskrcnn_benchmark and coco api dependencies
# make
python setup.py build_ext install
```

### Run
run through this file:  
samples/demo.ipynb

