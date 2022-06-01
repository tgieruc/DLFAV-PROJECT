# VITA, EPFL

import cv2
import socket
import sys
import numpy
import struct
import binascii

from PIL import Image
from detector import Detector
import argparse
import imutils

# image data
downscale = 4
width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels

# Set up detector
detector = Detector()
cam = cv2.VideoCapture(0)


while True:
    _, img = cam.read()
    if img is None:
        break
    img = imutils.resize(img, height=120)


    #######################
    # Detect
    #######################
    bbox, bbox_label = detector.forward(img)

    if bbox_label != [0.0]:
        print(f'detected : poi {bbox_label}')
        bbox_label = [1.0]
        bbox = bbox.astype(int)
        img = cv2.rectangle(img, [bbox[0]-5, bbox[1]-5], [bbox[0]+5, bbox[1]+5], (0,250,0),2)

    else:
        print("False")

    cv2.imshow("window", img)
    cv2.waitKey(1)