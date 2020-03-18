Obtain Models


Instructions
- run genmask.py to obtain a solo.json
- 


Demo for several approaches
0a) show stock coco mask drawing
0b) draw coco mask on a custom task - the solo synthetic dataset
1) genmask.py - synthetic dataset creation for detectron2, pixel-level segmentation, localizing logo
2) encoding an decoding uuid


python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
  MODEL.DEVICE cpu