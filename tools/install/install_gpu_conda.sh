#!/bin/bash
CONDA=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA/etc/profile.d/conda.sh

conda create -n darth python=3.7 -y
conda activate darth

# edit based on your CUDA version
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv (edit based on your CUDA version)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install mmdetection
pip install mmdet==2.25.1

# install mmtracking
pip install git+https://github.com/open-mmlab/mmtracking.git@904407e3d4e7147b45ffed422ba4c70829f5a224

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
