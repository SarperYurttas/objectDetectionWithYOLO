import numpy as np
import cv2
import mss
import os

LABELS_PATH = os.path.dirname(__file__) + '/coco.names'
LABELS = open(LABELS_PATH).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


class Frame:
    def __init__(self, source='webcam', screen_size=(800, 640)):
        self.source_name = source
        self.screen_size = (int(screen_size.split('x')[0]),
                            int(screen_size.split('x')[1])) if isinstance(screen_size, str) else screen_size
        self.source = cv2.VideoCapture(0) if source == 'webcam' else mss.mss()

    def get_frame(self):
        if self.source_name == 'webcam':
            _, image = self.source.read()
        elif self.source_name == 'screen':
            monitor = {"top": 0, "left": 0,
                       "width": self.screen_size[0], "height": self.screen_size[1]}
            image = np.array(self.source.grab(monitor))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
