#!/bin/bash
WORKSPACE=$1

python -m venv $WORKSPACE/venv/darth
export PYTHONPATH=$WORKSPACE/venv/darth/bin/python
source $WORKSPACE/venv/darth/bin/activate

# pip install --upgrade numpy

# install a pytorch version compatible with your gpu
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu102
# pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install --upgrade pip setuptools wheel
# install the latest mmcv (edit based on your CUDA version)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.12.0/index.html
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

# install mmdetection
pip install mmdet==2.25.1

# install mmtracking
pip install mmtrack==0.14.0

# install mmtracking dependencies
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/TAO-Dataset/tao.git

# install darth dependencies
pip install -r tools/install/requirements.txt
pip install git+https://github.com/SysCV/shift-dev.git
pip install git+https://github.com/bdd100k/bdd100k.git
pip install git+https://github.com/scalabel/scalabel.git@v0.3.0
pip install git+https://github.com/SysCV/tet.git#egg=teta\&subdirectory=teta

# setup darth
pip install -v -e .
