from .utils import Frame, COLORS, LABELS
from time import time
import numpy as np
import cv2
import os


def detect_from_image(image, model, layer_names):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416),	swapRB=True, crop=False)
    model.setInput(blob)
    layer_outputs = model.forward(layer_names)
    boxes, confidences, class_ids = [], [], []

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
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def yolov3(source='screen', model='yolov3-tiny', screen_size=(800, 640)):
    cfg = os.path.dirname(__file__) + f"/weights/{model}.cfg"
    weights = os.path.dirname(__file__) + f"/weights/{model}.weights"
    model = cv2.dnn.readNetFromDarknet(cfg, weights)

    layer_names = model.getLayerNames()
    layer_names = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    frame = Frame(source=source, screen_size=screen_size)

    while(True):
        loop_time = time()

        image = frame.get_frame()

        image = detect_from_image(image, model, layer_names)

        fps = "FPS: {:.1f}".format(1 / (time() - loop_time))
        cv2.putText(image, fps, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if source == 'screen':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", image)

        if cv2.waitKey(1) == ord("q"):
            if source == 'webcam':
                frame.source.release()
            cv2.destroyAllWindows()
            break
