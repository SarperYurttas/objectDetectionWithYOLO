from .yolov5 import yolov5
from .yolov3 import yolov3


def object_detection(source, model, screen_size):
    if 'yolov3' in model:
        yolov3(source=source, model=model, screen_size=screen_size)
    elif 'yolov5' in model:
        yolov5(source=source, model=model, screen_size=screen_size)
