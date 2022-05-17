#!/bin/bash

pip install -r requirements.txt
python Real-ESRGAN/setup.py develop
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth  -P Real-ESRGAN/experiments/pretrained_models
wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth
gdown 1iYxlkO_RZJ_6iL6wdIbfjl8VprLlFp1c

