#!/bin/bash
pip install torch
pip install -r requirements.txt

# Real esrgan                                                                                                                                                                                              
pip install -r Real-ESRGAN/requirements.txt
wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth  -P Real-ESRGAN/experiments/pretrained_models
wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P Real-ESRGAN/experiments/pretrained_models

# yolox weights                                                                                                                                                                                            
wget -nc https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_m.pth
# pose classifier weights                                                                                                                                                                                  
gdown 1iYxlkO_RZJ_6iL6wdIbfjl8VprLlFp1c
gdown 1heuooiFh4dMiYJCKcpB5nCXkj7lW7cRw
