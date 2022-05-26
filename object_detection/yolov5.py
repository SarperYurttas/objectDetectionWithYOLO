from .utils import Frame, COLORS, LABELS
from time import time
import cv2
import torch


def detect_from_image(model, image, size):
    result = model(image, size=size)
    preds = result.xywh[0].cpu().numpy()
    if len(preds) > 0:
        for i, pred in enumerate(preds):
            x, y, w, h, conf, label = pred
            x -= w/2
            y -= h/2
            color = COLORS[label.astype(int)].tolist()
            cv2.rectangle(image, (x.astype(int), y.astype(int)),
                          ((x + w).astype(int), (y + h).astype(int)), color, 3)

            text = "{}: {:.4f}".format(LABELS[label.astype(int)], conf)
            cv2.putText(image, text, (x.astype(int), y.astype(int) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def yolov5(source='screen', model='yolov5m', screen_size=(800, 640)):
    model = torch.hub.load('ultralytics/yolov5', model=model)
    frame = Frame(source=source, screen_size=screen_size)

    while(True):
        loop_time = time()
        image = frame.get_frame()

        detect_from_image(model, image, frame.screen_size[0])

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
