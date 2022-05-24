import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn.functional as F


import sys,os
sys.path.insert(0, os.path.join(os.path.abspath(''), "YOLOX"))

import cv2
import imutils

from tracker import Tracker
from pose_detector import PoseDetector
from sequence_detector import SequenceDetector
from pose_tracker import PoseTracker
from visualizer import Visualizer
from project_utils import identify_bbox
from pose_tracker import PoseTracker


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_c):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.box = torch.nn.Linear(n_hidden, n_output-1)   # output layer
        self.logit = torch.nn.Linear(n_hidden, 1)
        
        self.conv1 = torch.nn.Sequential(         # input shape (3, 80, 60)
            torch.nn.Conv2d(
                in_channels = n_c,            # input height
                out_channels = 8,             # n_filters
                kernel_size = 5,              # filter size
                stride = 2,                   # filter movement/step
                padding = 0,                  
            ),                              
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(kernel_size = 2),    
        )
        self.conv2 = torch.nn.Sequential(       
            torch.nn.Conv2d(in_channels = 8, 
                            out_channels = 16, 
                            kernel_size = 5, 
                            stride = 2, 
                            padding = 0),      
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),                
        )
        
        self.conv3 = torch.nn.Sequential(       
            torch.nn.Conv2d(in_channels = 16, 
                            out_channels = 8, 
                            kernel_size = 1, 
                            stride = 1, 
                            padding = 0),      
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),                
        )
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = feat.view(feat.size(0), -1)
        x2 = F.relu(self.hidden(feat))      # activation function for hidden layer
        
        out_box = F.relu(self.box(x2))            # linear output
        out_logit = torch.sigmoid(self.logit(x2))
        
        return out_box, out_logit
        
class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()
        # TODO: MEAN & STD
        self.mean = [[[[0.5548078,  0.56693329, 0.53457436]]]] 
        self.std = [[[[0.26367019, 0.26617227, 0.25692861]]]]
        self.img_size = 100 
        self.img_size_w = 80
        self.img_size_h = 60
        self.min_object_size = 10
        self.max_object_size = 40 
        self.num_objects = 1
        self.num_channels = 3
        self.model = Net(n_feature = 1632, n_hidden = 128, n_output = 5, n_c = 3)     # Defines the network #on a besoin de le modifier ?
        self.tracker = Tracker('person') #tracker a été update sur le git ? #Defines the used trackerz
        self.pose_detector = PoseDetector(n_feature=8, n_output=3, checkpoint_path="pose_classification_simplified.ckpt", simple_mode=True, model_based=False)#Defines the pose detection
        self.sequence_detector = SequenceDetector([1,2], 10) #defines the sequence detection that is used
        self.pose_tracker = PoseTracker(self.pose_detector, self.sequence_detector, self.tracker) #Defines our pose tracker
        self.upscaler=Upscaler(model_choice="realesrgan_s", enhance_face=False)

        self.poi = None #Person of Interest variable
        self.state = 0 #State of the FSM init
        self.check_poi=0 
`
    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def forward(self, img):   

        #En premier il faut pré-process les images que le robot nous envoie (avec le up-scaling)

        im=self.upscaler.enhance(img)

        #Ensuite il faut retourner les bbox et leur label au robot (sous le format [center_x,center_y,w,h])

        if self.state == 0:
           pred_bboxes, self.poi = self.pose_tracker.update(im)
           if self.poi != None:
               self.state = 1
               
        if self.state == 1:
            self.check_poi = 0
            pred_bboxes = np.array(self.tracker.update(im))
            for i in range(pred_bboxes.shape[0]) :
                if pred_bboxes[i,4] ==self.poi:
                    self.check_poi = 1
            if (self.check_poi == 0):
                self.poi = None
                self.state = 0

        

        #with torch.no_grad():

            #pred_y_label[0]=0
            #pred_y_label = pred_y_logit > 0.5 #il faut recup les id des bbox dans les tracker
            #pred_bboxes = pred_y_box * self.img_size
            #pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
            #for i in range(len(pred_y_box)):
                   #pred_y_label[i]=i
                
        if (pred_bboxes == Null):
            pred_bboxes[0]=[0,0,0,0]
            self.poi=0

        return pred_bboxes[self.poi], self.poi

