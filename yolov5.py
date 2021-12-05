import numpy as np
import cv2
import torch
import mss
import random
from time import time

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
colors = [(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for x in range(len(LABELS))]

sct = mss.mss() # this is for screen capture
monitor = {"top": 150, "left": 0, "width": 800, "height": 640} # this is for screen capture

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m') # for better mAP use either yolov5m, yolov5l, yolov5x but this will decrease fps.
#cam = cv2.VideoCapture(0) # this is for webcam

loop_time = time()
while(True):
    image = np.array(sct.grab(monitor))   # this is for screen capture
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # this is for screen capture
    #_,image = cam.read() # this is for webcam
    result = model(image, size = 800)
    
    preds = result.xywh[0].cpu().numpy()
    if len(preds) >0:
        for i, pred in enumerate(preds):
            x, y, w, h, conf, label = pred
            x -= w/2
            y -= h/2
            color = colors[label.astype(int)]
            cv2.rectangle(image, (x.astype(int), y.astype(int)), ((x + w).astype(int), (y + h).astype(int)), color, 3)
            text = "{}: {:.4f}".format(LABELS[label.astype(int)], conf)
            cv2.putText(image, text, (x.astype(int), y.astype(int) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    fps = "FPS: {:.1f}".format(1 / (time() - loop_time))
    loop_time = time()
    cv2.putText(image, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # this is for screen capture
    cv2.imshow("Image", image)

    if cv2.waitKey(1) == ord("q"):
        # cam.release() #this is for webcam
        cv2.destroyAllWindows()
        break