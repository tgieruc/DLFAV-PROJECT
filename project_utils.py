import numpy as np

def identify_bbox(bbox_target, bboxes_with_id):

    target_id = None
    if (bboxes_with_id is not None) and (bbox_target is not None):
        id = bboxes_with_id[:, 4]
        bboxes = bboxes_with_id[:, :4]
        bbox_target[2:] = bbox_target[2:] + bbox_target[2:]
        IoU = []
        for box in bboxes:
            IoU.append(bb_intersection_over_union(bbox_target, box))

        if max(IoU) > 0:
            target_id = id[np.argmax(IoU)]

    return target_id

def bb_intersection_over_union(boxA, boxB): #https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
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


