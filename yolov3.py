from time import time
import argparse
import numpy as np
import cv2
import mss

LABELS_PATH = "coco.names"
LABELS = open(LABELS_PATH).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def edge_detect(img):
    oimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(oimg, (5, 5), 1)
    edges = cv2.Canny(blur_gray, 50, 150)
    vertices = np.array([[10, 600], [10, 500], [300, 370],
                         [500, 370], [800, 500], [800, 600]])
    oimg = roi(edges, [vertices])
    return oimg


def draw_lines(image):
    gray = edge_detect(image)
    linesP = cv2.HoughLinesP(gray, 1, np.pi / 180, 50, None, 100, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]),
                     (0, 0, 255), 3, cv2.LINE_AA)
    return image


def detect_from_image(image, net, layer_names):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416),	swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)
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


def main(source='screen', weights='tiny', lines=False, screen_size=(800, 640)):
    net = cv2.dnn.readNetFromDarknet(f"weights/yolov3-{weights}.cfg", f"weights/yolov3-{weights}.weights")

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    frame = Frame(source=source, screen_size=screen_size)

    while(True):
        loop_time = time()

        image = frame.get_frame()

        image = draw_lines(image=image) if lines else image
        image = detect_from_image(image, net, layer_names)

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


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="screen")
    parser.add_argument('--weights', type=str, default="tiny")
    parser.add_argument('--lines', type=bool, default=False)
    parser.add_argument('--screensize', type=str, default="800x640")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(source=opt.source, weights=opt.weights, lines=opt.lines, screen_size=opt.screensize)
