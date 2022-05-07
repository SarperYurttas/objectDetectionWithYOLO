import argparse
from object_detection import object_detection


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="screen")
    parser.add_argument('--model', type=str, default="yolov5m")
    parser.add_argument('--screensize', type=str, default="800x640")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    object_detection(source=opt.source, model=opt.model,
                     screen_size=opt.screensize)
