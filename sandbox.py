import cv2
import numpy as np

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger

import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def readcam():
    """
    Example code that is ready to be incorporated, using the webcam, instead of a video file
    :return:
    """
    camera_resolution = (960., 1280.) #720 builtin 960 logitech
    stream = cv2.VideoCapture(1)
    a = 1
    while True:
        grabbed, frame = stream.read()
        if not grabbed:
            break

        output_image = frame #[:, :, ::-1]
        # do stuff in RGB, such as grab a predict frame for imshow, below
        # cv2.putText(output_image, "Demo", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 100, 0), 1)
        edges = cv2.Canny(output_image, 100, 200)
        cv2.imshow('', edges)#[:, :, ::-1])  # convert back to BGR for cv2
        if a == 50:
            cv2.imwrite('./bwout.jpg', edges)
        a += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def debug():
    img = cv2.imread('./bwout.jpg', cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    amax = np.argmax(np.argmax(im_bw, axis=0))

    cv2.imshow('thresh {}'.format(thresh), im_bw)
    cv2.waitKey()


# def pyimgsearch():
#     # load the image, convert it to grayscale, and blur it slightly
#     image = cv2.imread('./bwout.jpg')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     # threshold the image, then perform a series of erosions +
#     # dilations to remove any small regions of noise
#     thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
#     thresh = cv2.erode(thresh, None, iterations=2)
#     thresh = cv2.dilate(thresh, None, iterations=2)
#     # find contours in thresholded image, then grab the largest
#     # one
#     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     c = max(cnts, key=cv2.contourArea)

if __name__ == '__main__':
    assert torch.__version__ == '1.4.0', print(torch.__version__)
    setup_logger()

    im = cv2.imread("./images/manonhorse.jpg")
    # cv2.imshow('', im)
    # cv2.waitKey()
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    # pyimgsearch()
    # debug()
    # readcam()

