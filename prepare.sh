#!/usr/bin/bash

git submodule update --init --recursive
pip install -U "setuptools<70"
pip install gdown
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git
PYTORCH3D_NO_NINJA=1 pip install git+https://github.com/facebookresearch/pytorch3d.git # no ninja for build stability
cd third_party/kaolin && FORCE_CUDA=1 python setup.py develop && cd ../..

# Install FoundationPose
cd third_party/FoundationPose/mycpp
mkdir build && cd build
cmake .. && make -j 6
cd ../../bundlesdf/mycuda && pip install -e . && cd ../..
gdown 1jBx_FcI00Jb6k2DkgwF8eilkQsZZWoZp && unzip foundation_pose_ckpt.zip && rm foundation_pose_ckpt.zip
gdown 1f8pi8w3dopXHya3PfmM1DykgqC4AGkkO && unzip foundation_pose_demo_data.zip && rm foundation_pose_demo_data.zip
cd ../..
