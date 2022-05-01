import numpy as np

class PoseTracker(object):
    """Container for pose detector, sequence detector and tracker"""
    def __init__(self, pose_detector, sequence_detector, tracker):
        self.pose_detector = pose_detector
        self.sequence_detector = sequence_detector
        self.tracker = tracker

        self.current_id = None
    
    def update(self, im): 
        # Pose detector 
        bbox, pose = self.pose_detector.inference(im)
        # Tracker
        outputs = np.array(self.tracker.update(im))
        if len(outputs > 0):
            # Filter out IDs that are not in both tracker and pose detectors
            idt, idp = self._id_filtering(outputs, bbox)
            # Sequence detector
            id_t = self.sequence_detector.detection(pose[idp], idt)
            if id_t is not None:
                self.current_id = id_t
        return outputs, self.current_id

    def _id_filtering(self, outputs, pbbox):
        idt = []; idp = []
        for i in range(len(outputs)):
            id_p = self._identify_bbox(outputs[i], pbbox)
            if id_p is not None:
                idt.append(outputs[i,4])
                idp.append(id_p)
        return idt, idp

    def _identify_bbox(self, bbox_tracker, bboxes_pose):
        # returns pose_id, the id of the bbox pose where IoU with tracker bbox is max.
        pose_id = None
        if (len(bboxes_pose) != 0) and (len(bbox_tracker) != 0):
            id = np.arange(len(bboxes_pose))
            bboxes_pose[:,2:] = bboxes_pose[:,2:] + bboxes_pose[:,:2]
            IoU = []
            for box in bboxes_pose:
                IoU.append(self._bb_intersection_over_union(bbox_tracker[:4], box))
            if max(IoU) > 0:
                pose_id = id[np.argmax(IoU)]
        return pose_id

    def _bb_intersection_over_union(self, boxA, boxB): #https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou