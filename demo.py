import sys,os
sys.path.insert(0, os.path.join(os.path.abspath(''), "YOLOX"))


from tracker import Tracker
from detector import Detector
import imutils, argparse, cv2
import os
from glob import glob



def track_cap():
    cap = cv2.VideoCapture(0)
    tracker = Tracker()
    a = 0
    while True:
        
        _, im = cap.read()
        if im is None:
            break
        a += 1
        if a%10!=0:
            continue
        im = imutils.resize(im, height=500)
        image,_ = tracker.update(im)
       
 
        cv2.imshow('demo', image)
        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':


    track_cap()

        