import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import random
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


if __name__ == '__main__':

    # for d in ["train", "val"]:
    #     DatasetCatalog.register("./images/balloon/" + d, lambda d=d: get_balloon_dicts("./images/balloon/" + d))
    #     MetadataCatalog.get("./images/balloon/" + d).set(thing_classes=["balloon"])
    # balloon_metadata = MetadataCatalog.get("./images/balloon/train")

    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("solo_dataset", {}, "./solo.json", "./traindata/")
    from detectron2.data import MetadataCatalog

    MetadataCatalog.get("solo_dataset").thing_classes = ["solo"]
    dataset_dicts = DatasetCatalog.get("solo_dataset")

    import random

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1])

    # cfg = get_cfg()
    # cfg.merge_from_file("./configs/mask_rcnn_R_50_FPN_3x.yaml")  #./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
    # cfg.DATASETS.TRAIN = ("solo_dataset")
    # cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    #
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # # load weights
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    # # Set training data-set path
    # cfg.DATASETS.TEST = ("./images/balloon/val",)
    # # Create predictor (model for inference)
    # predictor = DefaultPredictor(cfg)
    #
    # dataset_dicts = get_balloon_dicts("./images/balloon/val")
    # for d in random.sample(dataset_dicts, 3):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=balloon_metadata,
    #                    scale=0.8,
    #                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    #                    )
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imshow('', v.get_image()[:, :, ::-1])
    #     cv2.waitKey()
