import cv2


class Visualizer(object):
    def __init__(self, env):
        self.env = env


    def _get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        return color

    def show(self, frame, bboxes=None, poi=None):
        if self.env == "colab":
            self._colab_show(frame, bboxes, poi)
        elif self.env == "local":
            self._local_show(frame, bboxes, poi)

    def _colab_show(self, frame, bboxes=None,  poi=None):
        pass

    def _local_show(self, frame, bboxes=None, poi=None):
        if bboxes is not None:
            if len(bboxes) > 0:
                ids = bboxes[:,4]
                H,W = frame.shape[:2]
                intbox = bboxes[:,:4].reshape(-1, 4).astype(int)
                # intbox[:,2:] = intbox[:,2:] + intbox[:, :2]
                intbox[intbox[:,0] < 0, 0] = 0
                intbox[intbox[:,2] < 0, 2] = 0
                intbox[intbox[:,1] > H - 1, 1] = H - 1
                intbox[intbox[:,3] > W - 1, 3] = W - 1
                intbox = intbox.reshape(-1,2,2)
                for i, box in enumerate(bboxes):
                    color = (255,0,0)
                    thickness = 3
                    if ids is not None:
                        obj_id = int(ids[i])
                        id_text = '{}'.format(int(obj_id))
                        color = self._get_color(abs(obj_id))
                        if poi is not None:
                            if poi == obj_id:
                                id_text = 'Person of Interest, ' + '{}'.format(int(obj_id))
                                color = (0,0,255)
                                thickness = 6
                        cv2.putText(frame, id_text, (intbox[i,0,0] , intbox[i,0,1]) , cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
                        cv2.rectangle(frame, intbox[i,0],  intbox[i,1], color, thickness)
                    else:
                        cv2.rectangle(frame, intbox[i,0],  intbox[i,1], color, 3)
        cv2.imshow("window", frame)
        cv2.waitKey(5)

