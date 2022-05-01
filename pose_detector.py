import torch
from torch.nn import functional as F
import numpy as np
import copy
import sys,os
# sys.path.insert(0, os.path.join(os.path.abspath(''), "openpifpaf/src/openpifpaf"))

import openpifpaf


class KeypointToPoseNet(torch.nn.Module):
    """Fully connected linear net to classify human pose from keypoints"""
    def __init__(self, n_feature, n_output):
        super(KeypointToPoseNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_feature, 32)
        self.fc3 = torch.nn.Linear(32, n_output)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class PoseDetector(object):
    """Given an image, it detects humans and their body position"""
    def __init__(self, n_feature, n_output, checkpoint_path, simple_mode=True, model_based=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.keypoint_to_pose_net = KeypointToPoseNet(n_feature=n_feature, n_output=n_output).cuda(device)
        self.keypoint_to_pose_net.load_state_dict(torch.load(checkpoint_path))

        self.openpifpaf_predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')

        self.simple_mode = simple_mode
        self.model_based = model_based
        self.keypoints = None
        self.keypoints_filtered = None
        self.openpifpaf_predictions = None
        self.pose_predictions = None
        self.centers = None
        self.bboxes = np.array([])
        self.scores = np.array([])
        self.tensor_prediction = None
        self.face_box = False
        self.face_tightness=1

    def inference(self, frame):
        #openpifpaf does prediction
        self.openpifpaf_predictions, _, _ = self.openpifpaf_predictor.numpy_image(frame)
        # get keypoints and bounding boxes
        self._get_keypoints()
        self._get_bboxes()
        # predicts the pose of each person
        self._keypoints_to_pose()

        return self.bboxes, self.pose_predictions


    def _get_keypoints(self):
        n_prediction = len(self.openpifpaf_predictions)
        self.keypoints = np.zeros((n_prediction, 17,3))
        for i, prediction in enumerate(self.openpifpaf_predictions):
            self.keypoints[i] = prediction.data



    def _keypoints_to_pose(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # filter keypoints
        if self.simple_mode:
            ind = np.arange(5,9)
        else:
            ind = np.arange(5,11)
        n_predictions = len(self.openpifpaf_predictions)
        self.pose_predictions = np.zeros(n_predictions, dtype=int)

        filtered_id = np.argwhere(self.keypoints[:,ind,2].min(1) > 0.5).flatten()
        if len(filtered_id) == 0:
            return False
        self.keypoints_filtered = copy.deepcopy(self.keypoints[filtered_id][:,ind,:2])


        # predict positions
        if self.model_based:
            lim1 = [3.3, 2.7, 2.3, 1.7, 3.2, 2.7, 3.2, 2.7] # angle limits for pose 1 (L)
            lim2 = [2.3, 1.7, 3.3, 2.7, 3.2, 2.7, 3.2, 2.7] # angle limits for pose 2 (invL)
            for i, p in enumerate(self.keypoints_filtered):
                if self.simple_mode:
                    v = [p[0]-p[1],p[2]-p[0],p[3]-p[1]]
                    v = np.array([v[j] / np.linalg.norm(v[j]) for j in range(len(v))])
                    a1 = np.pi - np.arccos(np.dot(v[1],v[0])) # rsh lsh lel
                    a2 = np.pi - np.arccos(np.dot(-v[2],v[0])) # lsh rsh rel
                    if p[0,1] < p[2,1]:
                        a1 = 2*np.pi - a1
                    if p[1,1] < p[3,1]:
                        a2 = 2*np.pi - a2
                    a = [a1, a2]
                    if a[0] < lim1[0] and a[0] > lim1[1] and \
                      a[1] < lim1[2] and a[1] > lim1[3]:
                        self.pose_predictions[filtered_id[i]] = 1
                    elif a[0] < lim2[0] and a[0] > lim2[1] and \
                      a[1] < lim2[2] and a[1] > lim2[3]:
                        self.pose_predictions[filtered_id[i]] = 2
                    else:
                      self.pose_predictions[filtered_id[i]] = 0
                else:
                    v = [p[0]-p[1],p[2]-p[0],p[4]-p[2],p[3]-p[1],p[5]-p[3]]
                    v = np.array([v[j] / np.linalg.norm(v[j]) for j in range(len(v))])
                    a1 = np.pi - np.arccos(np.dot(v[1],v[0])) # rsh lsh lel
                    a2 = np.pi - np.arccos(np.dot(-v[3],v[0])) # lsh rsh rel
                    a3 = np.pi - np.arccos(np.dot(v[2],v[1])) # lsh lel lwr
                    a4 = np.pi - np.arccos(np.dot(v[4],v[3])) # rsh rel rwr
                    if p[0,1] < p[2,1]:
                        a1 = 2*np.pi - a1
                    if p[1,1] < p[3,1]:
                        a2 = 2*np.pi - a2
                    if p[2,1] < p[4,1]:
                        a3 = 2*np.pi - a3
                    if p[3,1] < p[5,1]:
                        a4 = 2*np.pi - a4
                    a = [a1, a2, a3, a4]
                    if a[0] < lim1[0] and a[0] > lim1[1] and \
                      a[1] < lim1[2] and a[1] > lim1[3] and \
                      a[2] < lim1[4] and a[2] > lim1[5] and \
                      a[3] < lim1[6] and a[3] > lim1[7]:
                        self.pose_predictions[filtered_id[i]] = 1
                    elif a[0] < lim2[0] and a[0] > lim2[1] and \
                      a[1] < lim2[2] and a[1] > lim2[3]and \
                      a[2] < lim2[4] and a[2] > lim2[5] and \
                      a[3] < lim2[6] and a[3] > lim2[7]:
                        self.pose_predictions[filtered_id[i]] = 2
                    else:
                      self.pose_predictions[filtered_id[i]] = 0

        else:
          for i, keypoints in enumerate(self.keypoints_filtered.reshape(-1,8)):
              keypoints -= keypoints.mean(axis=0)
              keypoints /= keypoints.std(axis=0)
              keypoints = keypoints.reshape(1,-1)
              _, pred = torch.max(F.softmax(self.keypoint_to_pose_net(torch.from_numpy(keypoints.astype(np.float32)).cuda(device))).data, dim=1)
              self.pose_predictions[filtered_id[i]] = pred.int().cpu()

    def _get_bboxes(self):
        n_predictions = len(self.openpifpaf_predictions)
        self.bboxes = np.zeros((n_predictions, 4))
        self.scores = np.ones((n_predictions, 2))

        for i in range(n_predictions):
            # self.scores[i] = self.openpifpaf_predictions[i].score

            if (not (self.keypoints[i, :3, :2] == 0).any()) and self.face_box:
                face = self.keypoints[i, :5, :2]
                d_eyes = np.linalg.norm(face[1] - face[2])
                if (face[3] == 0.0).any() and (face[4] == 0.0).any():  # no ears
                    d_ears = 2 * d_eyes
                elif (face[3] == 0.0).any():  # no left ear
                    d_ears = np.linalg.norm(face[1] - face[4])
                elif (face[4] == 0.0).any():  # no right ear
                    d_ears = np.linalg.norm(face[2] - face[3])
                else:
                    d_ears = np.linalg.norm(face[3] - face[4])
                w = d_ears * self.face_tightness
                h = (d_ears + d_eyes) * self.face_tightness
                x = face[0, 0] - w / 2
                y = face[0, 1] - h / 2
                self.bboxes[i] = np.array([x, y, w, h])
            else:
                self.bboxes[i] = self.openpifpaf_predictions[i].bbox()

