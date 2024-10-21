#!/usr/bin/bash

git submodule update --init --recursive
pip install -U "setuptools<70"
pip install gdown
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/facebookresearch/pytorch3d.git # no ninja for build stability
cd third_party/kaolin
FORCE_CUDA=1 python setup.py develop
cd ../..

# Install FoundationPose
cd third_party/FoundationPose/mycpp
mkdir build
cd build
cmake .. && make -j$(nproc)
cd ../../bundlesdf/mycuda
pip install -e .
cd ../..
gdown 1jBx_FcI00Jb6k2DkgwF8eilkQsZZWoZp && unzip foundation_pose_ckpt.zip && rm foundation_pose_ckpt.zip
gdown 1f8pi8w3dopXHya3PfmM1DykgqC4AGkkO && unzip foundation_pose_demo_data.zip && rm foundation_pose_demo_data.zip
cd ../..

# Install MegaPose
cd third_party/megapose6d && pip install -e .
python -m megapose.scripts.download --megapose_models
cd ../..

# Install PoseEstimation
pip install -e .
