
https://github.com/foolwood/SiamMask

# Install
cd into the repo
> $ export SiamMask=$PWD
> $ export PYTHONPATH=$PWD:$PYTHONPATH

# Demo
cd $SiamMask/experiments/siammask
export PYTHONPATH=$PWD:$PYTHONPATH


* Official example
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json

* My
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_my_bottle.json --base_path ../../data/my_bottle
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_my_bottle.json --base_path ../../data/my_meter


## Others

* work with cpu:
https://github.com/foolwood/SiamMask/issues/15

## Errors during installation

* Torch not working with python 3.7:
https://github.com/HazyResearch/metal/issues/101
Use a specific version: torch==0.4.1.post2