from xml.etree.ElementInclude import include
import numpy as np
import time

# in: pose, bbox
# out: detected (bool), bbox 
class SequenceDetector(object):
    """Given poses and ids, it detects a sequence of pose"""
    def __init__(self, sequence, reset_time, simple_mode=True, model_based=False, face_bbox=False, face_tightness=1):
        self.sequence = sequence
        self.reset_time = reset_time
        self.decay_time = 5
        self.face_bbox = face_bbox
        self.face_tightness = face_tightness

        self.people = {}
        self.i_seq = np.array([]) # array containing position in sequence
        self.id_prev = np.array([]) # array containing tracker id of prediction
        self.t_prev = np.array([]) # array containing time since last corresponding pose detected in sequence

    def detection(self, pose, id_new): 
        id_out = None
        if len(self.people) == 0:
            self._initialize(id_new)
        else:
            id_out = self._update(pose, id_new)
        return id_out
            
    def _update_att(self, id_new):
        id_to_del = []
        # add new people
        for idt in id_new:
            if idt not in self.people:
                self.people[idt] = Person()
        # set visibility
        for idt in self.people:
            if idt not in id_new:
                self.people[idt].is_visible = False
                # ageism
                if time.time() - self.people[idt].t_last_seen > self.decay_time:
                    id_to_del.append(idt)
            else:
                self.people[idt].is_visible = True
                self.people[idt].t_last_seen = time.time()
        for idt in id_to_del:
            del(self.people[idt])
    def _initialize(self, id_new):
        if len(id_new) > 0:
            for i in id_new:
                self.people[i] = Person()

    def _reset(self):
        for i in self.people:
            self.people[i].reset()
    
    def _update(self, pose, id_new):
        self._update_att(id_new)
        for i in range(len(id_new)):
            idt = id_new[i]
            i_seq = int(self.people[idt].i_seq)
            # check pose
            if pose[i] != 0:
                if pose[i] == self.sequence[i_seq]: # reached next in sequence
                    self.people[idt].inc()
                    i_seq += 1
                    if i_seq >= len(self.sequence): # reached end of sequence
                        self._reset()
                        return idt
                elif i_seq >= 0 and pose[i] == self.sequence[i_seq-1]: # same pose
                    self.people[idt].t_prev = time.time()
                else: # different pose
                    if time.time() - self.people[idt].t_prev > self.reset_time:
                        self.people[idt].reset()
            else: # rien detected
                if time.time() - self.people[idt].t_prev > self.reset_time:
                        self.people[idt].reset()
        return None

class Person(object):
    def __init__(self):
        self.i_seq = 0
        self.t_prev = time.time()
        self.t_last_seen = time.time()
        self.is_visible = True
    
    def reset(self):
        self.i_seq = 0
        self.t_prev = time.time()

    def inc(self):
        self.i_seq += 1
        self.t_prev = time.time()
