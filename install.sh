#!/bin/bash

# Create a virtual environment
python3 -m venv hotd

# Activate the virtual environment
source hotd/bin/activate

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
git clone https://github.com/homm/pillow-lut-tools.git
mv pillow-lut-tools/* .
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P weights

pip install -r requirements_HOTD.txt

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -q mediapipe
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git

pip uninstall transformers -y
pip install transformers

pip uninstall opencv-python -y
pip uninstall opencv-python-headless -y
pip install opencv-python-headless

mkdir config_tests
mkdir outputs
mkdir all_skin_tones/