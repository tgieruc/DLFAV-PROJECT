import numpy as np
import time

# in: pose, bbox
# out: detected (bool), bbox 
class SequenceDetector(object):
    """Given poses and ids, it detects a sequence of pose"""
    def __init__(self, sequence, reset_time, simple_mode=True, model_based=False, face_bbox=False, face_tightness=1):
        self.sequence = sequence
        self.reset_time = reset_time
        self.face_bbox = face_bbox
        self.face_tightness = face_tightness

        self.i_seq = np.array([]) # array containing position in sequence
        self.id_prev = np.array([]) # array containing tracker id of prediction
        self.t_prev = np.array([]) # array containing time since last corresponding pose detected in sequence

    def detection(self, pose, id_new): 
        id_out = None
        if len(self.id_prev) == 0:
            self._reset(id_new)
        else:
            id_out = self._update(pose, id_new)
        return id_out
            
    def _update_att(self, id_new):
        # delete unecessary i_seq, id_prev if id_prev is not in pose_detector's id list
        self.id_prev, _, idx = np.intersect1d(id_new,self.id_prev,return_indices=True)
        if idx.size == 0:
            self.i_seq = np.array([0])
            self.t_prev = time.time()
        else:
            self.i_seq = self.i_seq[idx]
            self.t_prev = self.t_prev[idx]
        # add new ids
        if len(self.id_prev) < len(id_new):
            self.i_seq = np.append(self.i_seq, np.zeros(len(id_new) - len(self.id_prev)))
            self.t_prev = np.append(self.t_prev, np.repeat(time.time(),len(id_new) - len(self.id_prev)))
            self.id_prev = np.array(id_new)

    def _reset(self, id_new):
        if len(id_new) == 0:
            self.i_seq = np.array([0])
            self.id_prev = np.array([0])
            self.t_prev = np.array([time.time()])
        else:
            self.i_seq = np.zeros(len(id_new))
            self.id_prev = id_new
            self.t_prev = np.repeat(time.time(),len(id_new))
    
    def _update(self, pose, id_new):
        self._update_att(id_new)
        for i in range(len(self.id_prev)):
            # check pose
            if pose[i] != 0:
                if pose[i] == self.sequence[int(self.i_seq[i])]: # reached next in sequence
                    self.i_seq[i] += 1
                    self.t_prev[i] = time.time()
                    if int(self.i_seq[i]) >= len(self.sequence): # reached end of sequence
                        self._reset(id_new)
                        return id_new[i]
                elif self.i_seq[i] >= 0 and pose[i] == self.sequence[int(self.i_seq[i])-1]: # same pose
                    self.t_prev[i] = time.time()
                else: # different pose
                    if time.time() - self.t_prev[i] > self.reset_time:
                        self.i_seq[i] = 0
                        self.t_prev[i] = time.time()
            else: # rien detected
                if (time.time() - self.t_prev[i]) > self.reset_time:
                        self.i_seq[i] = 0
                        self.t_prev[i] = time.time()
        return None