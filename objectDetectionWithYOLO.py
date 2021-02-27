import numpy as np
import cv2
from time import time
import mss

def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(img,mask)
    return masked
def edge_detect(img):
    oimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(oimg,(5, 5),1)
    edges = cv2.Canny(blur_gray, 50, 150)
    vertices = np.array([[10,600], [10,500], [300,370], [500,370], [800,500], [800,600]])
    oimg = roi(edges,[vertices])
    return oimg

np.random.seed(42)
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

net = cv2.dnn.readNetFromDarknet("yolov3-spp.cfg", "yolov3-spp.weights")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

sct = mss.mss() # this is for screen capture
monitor = {"top": 150, "left": 0, "width": 800, "height": 640} # this is for screen capture

#cam = cv2.VideoCapture(0) # this is for webcam

loop_time = time()
while(True):
    image = np.array(sct.grab(monitor))   # this is for screen capture
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # this is for screen capture
    #_,image = cam.read() # this is for webcam

    gray = edge_detect(image)
    linesP = cv2.HoughLinesP(gray, 1, np.pi / 180, 50, None, 100, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    (H, W) = image.shape[:2]    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),	swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps = "FPS: {:.1f}".format(1 / (time() - loop_time))
    loop_time = time()
    cv2.putText(image, fps, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # this is for screen capture
    cv2.imshow("Image", image)

    if cv2.waitKey(1) == ord("q"):
        # cam.release() #this is for webcam
        cv2.destroyAllWindows()
        break