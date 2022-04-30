import numpy as np

boxes = np.array([[43,274,350,496,0], [291,  88, 653, 497,13]])
bbox_poi = np.array([380,219,1578,956])

def get_poi(bbox_poi, bboxes_with_id):
    poi_id = None
    if (bboxes_with_id is not None) and (bbox_poi is not None):
        id = bboxes_with_id[:, 4]
        bboxes = bboxes_with_id[:, :4]
        bbox_poi[2:] = bbox_poi[2:] + bbox_poi[2:]
        IoU = []
        for box in bboxes:
            IoU.append(bb_intersection_over_union(bbox_poi, box))

        if max(IoU) > 0:
            poi_id = id[np.argmax(IoU)]


    return poi_id

def bb_intersection_over_union(boxA, boxB):
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


print(get_poi(bbox_poi, boxes))